//! `EvictionPolicy` trait — the contract every Rust policy implements.
//!
//! The shape mirrors `src/policies/base.py`. Each policy:
//!   - reads a slice of `BlockMeta` (the eviction candidates)
//!   - returns the *index* of the candidate to evict
//!   - optionally takes feedback about whether an eviction was good
//!
//! `score()` is used by the CACHEUS ensemble — each expert produces
//! per-candidate scores, the ensemble combines them, and the highest
//! weighted score wins.

use crate::block::BlockMeta;

pub trait EvictionPolicy {
    /// Pick which candidate to evict. Returns the index into `candidates`.
    /// Panics if `candidates` is empty (callers should filter).
    fn select_victim(&mut self, candidates: &[BlockMeta]) -> usize;

    /// Per-candidate scores in [0, 1] where higher = more evictable.
    /// Default: select_victim's choice gets 1.0, others 0.0.
    fn score(&mut self, candidates: &[BlockMeta]) -> alloc::vec::Vec<f32> {
        use alloc::vec;
        let n = candidates.len();
        if n == 0 {
            return vec![];
        }
        let mut scores = vec![0.0; n];
        let victim = self.select_victim(candidates);
        scores[victim] = 1.0;
        scores
    }

    /// Optional feedback about a previous eviction.
    /// `was_fault = true` means the evicted block was re-accessed (bad).
    fn update_feedback(&mut self, _block_id: u32, _was_fault: bool) {
        // No-op for non-adaptive policies (LRU, LFU, SLM-Heuristic).
    }

    /// Reset internal state for a new simulation run.
    fn reset(&mut self) {}

    fn name(&self) -> &'static str;
}

extern crate alloc;
