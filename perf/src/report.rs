use core::fmt;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Measurements {
    pub name: String,
    pub cycles: usize,
    pub core_time: Duration,
    pub compress_time: Duration,
}

impl Measurements {
    pub fn new(name: &str, cycles: usize, core_time: Duration, compress_time: Duration) -> Self {
        Self {
            name: name.to_string(),
            cycles,
            core_time,
            compress_time,
        }
    }
}

impl fmt::Display for Measurements {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_time = self.core_time + self.compress_time;
        let khz = self.cycles as f64 / (total_time.as_secs_f64() * 1e3);
        let compress_fraction = self.compress_time.as_secs_f64() / total_time.as_secs_f64();

        writeln!(
            f,
            "{:<15} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}",
            "Program Name",
            "Cycles",
            "kHz",
            "Core Time (s)",
            "Compress Time (s)",
            "Total Time (s)",
            "Compress Fraction (%)"
        )?;
        writeln!(
            f,
            "{:<15} | {:<10} | {:<10.2} | {:<10.2} | {:<10.2} | {:<10.2} | {:<10.2}",
            self.name,
            self.cycles,
            khz,
            self.core_time.as_secs_f64(),
            self.compress_time.as_secs_f64(),
            total_time.as_secs_f64(),
            compress_fraction
        )
    }
}
