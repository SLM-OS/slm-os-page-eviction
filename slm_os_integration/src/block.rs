//! Block metadata and pool types — mirrors `src/simulator/block.py`.

/// Which memory pool a block belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PoolType {
    /// Read-only model weights (shared across inferences).
    Weight = 0,
    /// Read-write inference workspace (per-task).
    Workspace = 1,
}

/// Per-block metadata used by every eviction policy.
///
/// Mirrors `BlockMeta` in `src/simulator/block.py` — only the fields the
/// Rust policies actually read are included (the simulator carries more
/// state for trace recording, but eviction decisions don't need it).
#[derive(Debug, Clone, Copy)]
pub struct BlockMeta {
    pub block_id: u32,
    pub pool_type: PoolType,
    pub model_id: u8,
    pub layer_idx: i16,
    pub last_access_time: u64,
    pub load_time: u64,
    pub access_count: u32,
    pub ref_count: u8,
    pub gpu_mapped: bool,
    pub is_dirty: bool,
    pub model_priority: u8,
}

/// Flat 27-feature vector consumed by the trained ML policies.
///
/// The layout matches `FeatureConfig.feature_names` in
/// `src/features/extractor.py` — see `docs/features.md` for indices.
pub type BlockFeatures = [f32; 27];
