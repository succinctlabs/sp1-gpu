use serde::{ser::SerializeStruct, Serialize};
use sp1_core_executor::RiscvAirId;

use enum_map::EnumMap;

#[derive(Debug, Clone)]
pub struct ShardWithTime {
    pub shard: Shard,
    pub core_proving_time_ns: u64,
}

impl Serialize for ShardWithTime {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut row = serializer.serialize_struct("Outer", 3)?;
        row.serialize_field("program", &self.shard.program)?;
        row.serialize_field("shard_index", &self.shard.shard_index)?;
        row.serialize_field("core_proving_time_ns", &self.core_proving_time_ns)?;
        for (k, v) in &self.shard.shape {
            row.serialize_field(k.as_str(), v)?;
        }
        row.end()
    }
}

#[derive(Debug, Clone)]
pub struct Shard {
    pub program: String,
    pub shard_index: usize,
    pub shape: EnumMap<RiscvAirId, usize>, // EnumMap<RiscvAirId, usize>,
}

impl Serialize for Shard {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut row = serializer.serialize_struct("Outer", 3)?;
        row.serialize_field("program", &self.program)?;
        row.serialize_field("shard_index", &self.shard_index)?;
        for (k, v) in &self.shape {
            row.serialize_field(k.as_str(), v)?;
        }
        row.end()
    }
}
