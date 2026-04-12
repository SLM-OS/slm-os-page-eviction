//! Parity tests: each Rust policy must produce the same decisions as
//! its Python counterpart on the same inputs.
//!
//! These tests use hand-crafted candidate sets that exercise the same
//! tie-breaking edges checked in `tests/test_policies.py`.

use slm_os_eviction::{
    BlockMeta, CacheusSelector, EvictionPolicy, LfuPolicy, LruPolicy,
    PoolType, SlmHeuristicPolicy,
};

fn make_block(
    id: u32,
    last_access: u64,
    access_count: u32,
    pool_type: PoolType,
    model_id: u8,
) -> BlockMeta {
    BlockMeta {
        block_id: id,
        pool_type,
        model_id,
        layer_idx: 0,
        last_access_time: last_access,
        load_time: 0,
        access_count,
        ref_count: 0,
        gpu_mapped: false,
        is_dirty: false,
        model_priority: 0,
    }
}

#[test]
fn lru_evicts_oldest() {
    // Mirrors test_policies.py::TestLRUPolicy::test_evicts_oldest
    let candidates = [
        make_block(0, 100, 0, PoolType::Weight, 0),
        make_block(1, 50, 0, PoolType::Weight, 0),
        make_block(2, 200, 0, PoolType::Weight, 0),
    ];
    let mut policy = LruPolicy::new();
    let victim = policy.select_victim(&candidates);
    assert_eq!(candidates[victim].last_access_time, 50);
}

#[test]
fn lru_single_candidate() {
    let candidates = [make_block(0, 100, 0, PoolType::Weight, 0)];
    let mut policy = LruPolicy::new();
    assert_eq!(policy.select_victim(&candidates), 0);
}

#[test]
fn lru_score_monotonic() {
    let candidates = [
        make_block(0, 100, 0, PoolType::Weight, 0),
        make_block(1, 50, 0, PoolType::Weight, 0),
        make_block(2, 200, 0, PoolType::Weight, 0),
    ];
    let mut policy = LruPolicy::new();
    let scores = policy.score(&candidates);
    // Oldest (index 1, time=50) has highest score
    assert!(scores[1] > scores[0]);
    assert!(scores[0] > scores[2]);
}

#[test]
fn lfu_evicts_least_accessed() {
    // Mirrors test_policies.py::TestLFUPolicy::test_evicts_least_accessed
    let candidates = [
        make_block(0, 0, 10, PoolType::Weight, 0),
        make_block(1, 0, 1, PoolType::Weight, 0),
        make_block(2, 0, 5, PoolType::Weight, 0),
    ];
    let mut policy = LfuPolicy::new();
    let victim = policy.select_victim(&candidates);
    assert_eq!(candidates[victim].access_count, 1);
}

#[test]
fn lfu_breaks_ties_by_lru() {
    // Mirrors test_policies.py::TestLFUPolicy::test_breaks_ties_by_lru
    let candidates = [
        make_block(0, 200, 1, PoolType::Weight, 0),
        make_block(1, 100, 1, PoolType::Weight, 0),
        make_block(2, 300, 5, PoolType::Weight, 0),
    ];
    let mut policy = LfuPolicy::new();
    let victim = policy.select_victim(&candidates);
    assert_eq!(candidates[victim].last_access_time, 100);
}

#[test]
fn slm_heuristic_evicts_workspace_first() {
    // Mirrors test_policies.py::TestSLMHeuristicPolicy::test_evicts_workspace_first
    let candidates = [
        make_block(0, 100, 0, PoolType::Weight, 0),
        make_block(1, 200, 0, PoolType::Workspace, 0),
        make_block(2, 50, 0, PoolType::Weight, 0),
    ];
    let mut policy = SlmHeuristicPolicy::new();
    let victim = policy.select_victim(&candidates);
    assert_eq!(candidates[victim].pool_type, PoolType::Workspace);
}

#[test]
fn slm_heuristic_evicts_inactive_models_before_active() {
    // Mirrors test_policies.py::TestSLMHeuristicPolicy
    // ::test_evicts_inactive_models_before_active
    let candidates = [
        make_block(0, 100, 0, PoolType::Weight, 0),
        make_block(1, 200, 0, PoolType::Weight, 1),
        make_block(2, 50, 0, PoolType::Weight, 0),
    ];
    let mut policy = SlmHeuristicPolicy::new();

    // Mark model 0 as active; only model 1 (block 1) is inactive
    let mut active = alloc::collections::BTreeMap::new();
    active.insert(0, 1);
    policy.set_active_inferences(active);

    let victim = policy.select_victim(&candidates);
    assert_eq!(candidates[victim].model_id, 1);
}

#[test]
fn cacheus_initial_weights_uniform() {
    let experts: Vec<Box<dyn EvictionPolicy + Send>> = vec![
        Box::new(LruPolicy::new()),
        Box::new(LfuPolicy::new()),
    ];
    let cacheus = CacheusSelector::new(experts, 0.4, 200);
    let weights = cacheus.weights();
    assert_eq!(weights.len(), 2);
    assert!((weights[0] - 0.5).abs() < 1e-6);
    assert!((weights[1] - 0.5).abs() < 1e-6);
}

#[test]
fn cacheus_weights_sum_to_one_after_updates() {
    let experts: Vec<Box<dyn EvictionPolicy + Send>> = vec![
        Box::new(LruPolicy::new()),
        Box::new(LfuPolicy::new()),
    ];
    let mut cacheus = CacheusSelector::new(experts, 0.3, 100);

    let candidates = [
        make_block(0, 100, 0, PoolType::Weight, 0),
        make_block(1, 50, 0, PoolType::Weight, 0),
        make_block(2, 200, 0, PoolType::Weight, 0),
    ];
    let victim = cacheus.select_victim(&candidates);
    cacheus.update_feedback(candidates[victim].block_id, true);

    let total: f32 = cacheus.weights().iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "weights sum to {}", total);
}

#[test]
fn cacheus_reset_restores_uniform_weights() {
    let experts: Vec<Box<dyn EvictionPolicy + Send>> = vec![
        Box::new(LruPolicy::new()),
        Box::new(LfuPolicy::new()),
    ];
    let mut cacheus = CacheusSelector::new(experts, 0.5, 50);

    let candidates = [
        make_block(0, 100, 5, PoolType::Weight, 0),
        make_block(1, 50, 1, PoolType::Weight, 0),
        make_block(2, 200, 10, PoolType::Weight, 0),
    ];
    for _ in 0..10 {
        let v = cacheus.select_victim(&candidates);
        cacheus.update_feedback(candidates[v].block_id, true);
    }

    cacheus.reset();
    for &w in cacheus.weights() {
        assert!((w - 0.5).abs() < 1e-6, "weight {} != 0.5 after reset", w);
    }
}

#[test]
fn cacheus_min_weight_keeps_experts_recoverable() {
    // Even after many penalties, weights stay strictly positive.
    let experts: Vec<Box<dyn EvictionPolicy + Send>> = vec![
        Box::new(LruPolicy::new()),
        Box::new(LfuPolicy::new()),
    ];
    let mut cacheus = CacheusSelector::new(experts, 0.9, 100);

    let candidates = [
        make_block(0, 100, 5, PoolType::Weight, 0),
        make_block(1, 50, 1, PoolType::Weight, 0),
        make_block(2, 200, 10, PoolType::Weight, 0),
    ];
    for _ in 0..50 {
        let v = cacheus.select_victim(&candidates);
        cacheus.update_feedback(candidates[v].block_id, true);
    }

    for &w in cacheus.weights() {
        assert!(w > 0.0, "expert weight collapsed to {}", w);
    }
}

extern crate alloc;
