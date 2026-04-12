//! Hand-tuned SLM-OS eviction policy.
//!
//! Mirrors `src/policies/slm_heuristic.py` with the same priority cascade:
//!   1. Evict workspace blocks before weights (cheaper to recreate).
//!   2. Within workspace, evict LRU.
//!   3. Evict weights from models with no active inferences.
//!   4. Within inactive model weights, evict LRU.
//!   5. Fallback: evict LRU among all remaining candidates.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::block::{BlockMeta, PoolType};
use crate::eviction_policy::EvictionPolicy;

extern crate alloc;

#[derive(Default)]
pub struct SlmHeuristicPolicy {
    /// model_id → number of active inference tasks for that model.
    active_inferences: BTreeMap<u8, u32>,
}

impl SlmHeuristicPolicy {
    pub fn new() -> Self {
        Self { active_inferences: BTreeMap::new() }
    }

    /// Update the active-inference table (called by the runtime when an
    /// inference starts or finishes).
    pub fn set_active_inferences(&mut self, active: BTreeMap<u8, u32>) {
        self.active_inferences = active;
    }

    fn is_active(&self, model_id: u8) -> bool {
        self.active_inferences.get(&model_id).copied().unwrap_or(0) > 0
    }

    /// Return the original index of the LRU block among the supplied
    /// (original_index, block) pairs.
    fn pick_lru(indexed: &[(usize, &BlockMeta)]) -> usize {
        debug_assert!(!indexed.is_empty());
        let mut victim_idx = indexed[0].0;
        let mut min_time = indexed[0].1.last_access_time;
        for (idx, block) in indexed.iter().skip(1) {
            if block.last_access_time < min_time {
                min_time = block.last_access_time;
                victim_idx = *idx;
            }
        }
        victim_idx
    }
}

impl EvictionPolicy for SlmHeuristicPolicy {
    fn select_victim(&mut self, candidates: &[BlockMeta]) -> usize {
        debug_assert!(!candidates.is_empty(), "SLM select_victim on empty list");

        // Priority 1: workspace blocks (cheapest to recreate)
        let workspace: Vec<(usize, &BlockMeta)> = candidates
            .iter()
            .enumerate()
            .filter(|(_, b)| b.pool_type == PoolType::Workspace)
            .collect();
        if !workspace.is_empty() {
            return Self::pick_lru(&workspace);
        }

        // Priority 2: weights from models with no active inferences
        let inactive: Vec<(usize, &BlockMeta)> = candidates
            .iter()
            .enumerate()
            .filter(|(_, b)| !self.is_active(b.model_id))
            .collect();
        if !inactive.is_empty() {
            return Self::pick_lru(&inactive);
        }

        // Priority 3: LRU over all candidates
        let all: Vec<(usize, &BlockMeta)> = candidates.iter().enumerate().collect();
        Self::pick_lru(&all)
    }

    fn score(&mut self, candidates: &[BlockMeta]) -> Vec<f32> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let mut min_time = candidates[0].last_access_time;
        let mut max_time = candidates[0].last_access_time;
        for c in &candidates[1..] {
            if c.last_access_time < min_time { min_time = c.last_access_time; }
            if c.last_access_time > max_time { max_time = c.last_access_time; }
        }
        let span = if max_time == min_time { 1 } else { max_time - min_time };

        candidates
            .iter()
            .map(|b| {
                let mut s = 0.0_f32;
                if b.pool_type == PoolType::Workspace { s += 0.6; }
                if !self.is_active(b.model_id) { s += 0.3; }
                s += 0.1 * (max_time - b.last_access_time) as f32 / span as f32;
                s
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "SLM-Heuristic"
    }
}
