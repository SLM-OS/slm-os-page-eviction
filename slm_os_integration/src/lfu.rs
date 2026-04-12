//! Least-Frequently-Used eviction.
//!
//! Mirrors `src/policies/lfu.py`: pick the candidate with the smallest
//! `access_count`; ties are broken by older `last_access_time` (LRU
//! within the tied set).

use alloc::vec::Vec;

use crate::block::BlockMeta;
use crate::eviction_policy::EvictionPolicy;

extern crate alloc;

#[derive(Default)]
pub struct LfuPolicy;

impl LfuPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl EvictionPolicy for LfuPolicy {
    fn select_victim(&mut self, candidates: &[BlockMeta]) -> usize {
        debug_assert!(!candidates.is_empty(), "LFU select_victim on empty list");
        let mut victim = 0;
        let mut min_count = candidates[0].access_count;
        for (i, c) in candidates.iter().enumerate().skip(1) {
            if c.access_count < min_count {
                min_count = c.access_count;
                victim = i;
            } else if c.access_count == min_count
                && c.last_access_time < candidates[victim].last_access_time
            {
                victim = i;
            }
        }
        victim
    }

    fn score(&mut self, candidates: &[BlockMeta]) -> Vec<f32> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let mut min_count = candidates[0].access_count;
        let mut max_count = candidates[0].access_count;
        for c in &candidates[1..] {
            if c.access_count < min_count { min_count = c.access_count; }
            if c.access_count > max_count { max_count = c.access_count; }
        }
        let span = if max_count == min_count { 1 } else { max_count - min_count };
        candidates
            .iter()
            .map(|c| (max_count - c.access_count) as f32 / span as f32)
            .collect()
    }

    fn name(&self) -> &'static str {
        "LFU"
    }
}
