use core::fmt;
use csv::Writer;
use std::{error::Error, time::Duration};

#[derive(Debug, Clone)]
pub struct Measurement {
    pub name: String,
    pub cycles: usize,
    pub num_shards: usize,
    pub core_time: Duration,
    pub compress_time: Duration,
}

pub fn write_measurements_to_csv(
    measurements: &[Measurement],
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(filename)?;
    wtr.write_record([
        "Program Name",
        "Cycles",
        "Shards",
        "kHz",
        "Core Time (s)",
        "Compress Time (s)",
        "Total Time (s)",
        "Compress Fraction (%)",
    ])?;

    for measurement in measurements {
        let record = measurement.to_csv_record();
        wtr.serialize(record)?;
    }

    wtr.flush()?;
    Ok(())
}

impl Measurement {
    pub fn new(
        name: &str,
        cycles: usize,
        num_shards: usize,
        core_time: Duration,
        compress_time: Duration,
    ) -> Self {
        Self {
            name: name.to_string(),
            cycles,
            num_shards,
            core_time,
            compress_time,
        }
    }

    fn to_csv_record(&self) -> (String, usize, usize, f64, f64, f64, f64, f64) {
        let total_time = self.core_time + self.compress_time;
        let khz = self.cycles as f64 / (total_time.as_secs_f64() * 1e3);
        let compress_fraction =
            (self.compress_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0;

        (
            self.name.clone(),
            self.cycles,
            self.num_shards,
            khz,
            self.core_time.as_secs_f64(),
            self.compress_time.as_secs_f64(),
            total_time.as_secs_f64(),
            compress_fraction,
        )
    }
}

impl fmt::Display for Measurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_time = self.core_time + self.compress_time;
        let khz = self.cycles as f64 / (total_time.as_secs_f64() * 1e3);
        let compress_fraction = self.compress_time.as_secs_f64() / total_time.as_secs_f64();

        writeln!(
            f,
            "{:<15} | {:<10} | {:<10} | {:<10} | {:<15} | {:<15} | {:<15} | {:<20}",
            "Program Name",
            "Cycles",
            "Shards",
            "kHz",
            "Core Time (s)",
            "Compress Time (s)",
            "Total Time (s)",
            "Compress Fraction (%)"
        )?;
        writeln!(
            f,
            "{:<15} | {:<10} | {:<10} | {:<10.2} | {:<15.2} | {:<15.2} | {:<15.2} | {:<20.2}",
            self.name,
            self.cycles,
            self.num_shards,
            khz,
            self.core_time.as_secs_f64(),
            self.compress_time.as_secs_f64(),
            total_time.as_secs_f64(),
            compress_fraction
        )
    }
}
