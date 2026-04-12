//! CACHEUS-style adaptive expert selector.
//!
//! Mirrors `src/policies/cacheus.py` after the Phase 5 fixes:
//!   - `EvictionRecord` stores the ensemble's actual choice, not the
//!     last expert's choice.
//!   - On `was_fault=true` (evicted block was re-accessed), agreeing
//!     experts are penalized and disagreeing ones get a small reward.
//!   - On `was_fault=false`, agreeing experts are rewarded.
//!   - Weights are floored at `min_weight` and renormalized so no expert
//!     is permanently silenced.
//!
//! The Phase 5 sweep showed the `ml_only` pool (XGBoost + MLP, 2
//! experts) wins at 0.212 mean normalized fault rate — adding classical
//! experts dilutes the ensemble. The recommended runtime config is
//! lr=0.4, window=200.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::block::BlockMeta;
use crate::eviction_policy::EvictionPolicy;

extern crate alloc;

/// Records one ensemble decision so feedback can later reward/penalize
/// the experts that agreed with it.
#[derive(Clone)]
struct EvictionRecord {
    victim_block_id: u32,
    expert_choices: Vec<usize>, // per-expert candidate index
    ensemble_choice: usize,
}

/// Weighted ensemble of `EvictionPolicy` experts with online weight updates.
pub struct CacheusSelector {
    experts: Vec<alloc::boxed::Box<dyn EvictionPolicy + Send>>,
    weights: Vec<f32>,
    learning_rate: f32,
    min_weight: f32,
    window_size: usize,
    history: VecDeque<EvictionRecord>,
    expert_faults: Vec<u32>,
    expert_decisions: Vec<u32>,
}

impl CacheusSelector {
    /// Build a selector with uniform initial weights.
    ///
    /// The Phase 5 sweep recommends `learning_rate=0.4`, `window_size=200`
    /// for the `ml_only` (XGBoost + MLP) pool.
    pub fn new(
        experts: Vec<alloc::boxed::Box<dyn EvictionPolicy + Send>>,
        learning_rate: f32,
        window_size: usize,
    ) -> Self {
        let n = experts.len();
        assert!(n > 0, "CACHEUS requires at least one expert");
        let weights = alloc::vec![1.0 / n as f32; n];
        Self {
            experts,
            weights,
            learning_rate,
            min_weight: 0.01,
            window_size,
            history: VecDeque::with_capacity(window_size),
            expert_faults: alloc::vec![0; n],
            expert_decisions: alloc::vec![0; n],
        }
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn expert_names(&self) -> Vec<&'static str> {
        self.experts.iter().map(|e| e.name()).collect()
    }

    /// Apply multiplicative weight update for one feedback signal.
    fn update_weights(&mut self, record: &EvictionRecord, was_fault: bool) {
        for (i, &expert_choice) in record.expert_choices.iter().enumerate() {
            self.expert_decisions[i] += 1;
            let agreed = expert_choice == record.ensemble_choice;
            if was_fault {
                if agreed {
                    self.weights[i] *= 1.0 - self.learning_rate;
                    self.expert_faults[i] += 1;
                } else {
                    self.weights[i] *= 1.0 + 0.5 * self.learning_rate;
                }
            } else if agreed {
                self.weights[i] *= 1.0 + self.learning_rate;
            }
        }

        // Floor + renormalize.
        let mut total = 0.0_f32;
        for w in self.weights.iter_mut() {
            if *w < self.min_weight {
                *w = self.min_weight;
            }
            total += *w;
        }
        if total > 0.0 {
            for w in self.weights.iter_mut() {
                *w /= total;
            }
        }
    }
}

impl EvictionPolicy for CacheusSelector {
    fn select_victim(&mut self, candidates: &[BlockMeta]) -> usize {
        debug_assert!(!candidates.is_empty(), "CACHEUS select_victim on empty list");

        let n = candidates.len();
        let mut combined = alloc::vec![0.0_f32; n];
        let mut expert_choices = alloc::vec::Vec::with_capacity(self.experts.len());

        for (i, expert) in self.experts.iter_mut().enumerate() {
            let scores = expert.score(candidates);
            let mut best_idx = 0;
            let mut best_score = f32::MIN;
            for (j, &s) in scores.iter().enumerate() {
                combined[j] += self.weights[i] * s;
                if s > best_score {
                    best_score = s;
                    best_idx = j;
                }
            }
            expert_choices.push(best_idx);
        }

        let mut victim = 0;
        let mut max_combined = combined[0];
        for (j, &s) in combined.iter().enumerate().skip(1) {
            if s > max_combined {
                max_combined = s;
                victim = j;
            }
        }

        // Drop the oldest record if at capacity.
        if self.history.len() == self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(EvictionRecord {
            victim_block_id: candidates[victim].block_id,
            expert_choices,
            ensemble_choice: victim,
        });

        victim
    }

    fn update_feedback(&mut self, block_id: u32, was_fault: bool) {
        // Search history newest-to-oldest for the matching eviction.
        let found = self
            .history
            .iter()
            .rev()
            .find(|r| r.victim_block_id == block_id)
            .cloned();
        if let Some(record) = found {
            self.update_weights(&record, was_fault);
        }
    }

    fn reset(&mut self) {
        let n = self.experts.len();
        for w in self.weights.iter_mut() {
            *w = 1.0 / n as f32;
        }
        self.history.clear();
        for v in self.expert_faults.iter_mut() { *v = 0; }
        for v in self.expert_decisions.iter_mut() { *v = 0; }
        for e in self.experts.iter_mut() {
            e.reset();
        }
    }

    fn name(&self) -> &'static str {
        "CACHEUS"
    }
}
