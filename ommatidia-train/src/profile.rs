//! Non-overlapping host wall times around the existing synchronous training loop.
use std::{io::Write, time::Instant};

#[derive(Clone, Copy)]
pub enum Stage {
    ParameterSync,
    FrameDecode,
    WarmupAdvance,
    SlotAdvance,
    ReadPrepared,
    Feed,
    StepWait,
    LossReadback,
}

const NAMES: [&str; 8] = [
    "parameter_sync",
    "frame_decode",
    "warmup_advance",
    "slot_advance",
    "read_prepared",
    "feed",
    "step_wait",
    "loss_readback",
];

#[derive(Default)]
pub struct Update {
    seconds: [f64; 8],
    total_seconds: f64,
}

impl Update {
    pub fn record(&mut self, stage: Stage, start: Instant) {
        self.seconds[stage as usize] += start.elapsed().as_secs_f64();
    }

    pub fn finish(&mut self, start: Instant) {
        self.total_seconds = start.elapsed().as_secs_f64();
    }

    pub fn write_header(out: &mut impl Write) -> std::io::Result<()> {
        writeln!(out, "update,wall_seconds,{},other", NAMES.join(","))
    }

    pub fn write(&self, out: &mut impl Write, update: usize) -> std::io::Result<()> {
        write!(out, "{update},{:.9}", self.total_seconds)?;
        for seconds in self.seconds {
            write!(out, ",{seconds:.9}")?;
        }
        writeln!(out, ",{:.9}", self.other())
    }

    fn other(&self) -> f64 {
        (self.total_seconds - self.seconds.iter().sum::<f64>()).max(0.0)
    }
}

#[derive(Default)]
pub struct TrainingProfile {
    updates: usize,
    total: Update,
}

impl TrainingProfile {
    pub fn add(&mut self, update: &Update) {
        self.updates += 1;
        self.total.total_seconds += update.total_seconds;
        for (total, time) in self.total.seconds.iter_mut().zip(update.seconds) {
            *total += time;
        }
    }

    pub fn report(&self) -> serde_json::Value {
        if self.updates == 0 {
            return serde_json::Value::Null;
        }
        let wall = self.total.total_seconds;
        let stages: Vec<_> = NAMES
            .into_iter()
            .chain(["other"])
            .zip(self.total.seconds.into_iter().chain([self.total.other()]))
            .map(|(name, seconds)| {
                serde_json::json!({
                    "stage": name, "seconds": seconds,
                    "milliseconds_per_update": seconds * 1000.0 / self.updates as f64,
                    "fraction": seconds / wall,
                })
            })
            .collect();
        serde_json::json!({
            "updates": self.updates, "wall_seconds": wall,
            "updates_per_second": self.updates as f64 / wall,
            "stages": stages,
            "r1_step_wait_over_half": self.total.seconds[Stage::StepWait as usize] > 0.5 * wall,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_is_additive_and_accounts_for_unattributed_time() {
        let update = Update {
            seconds: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 60.0, 7.0],
            total_seconds: 100.0,
        };
        let mut profile = TrainingProfile::default();
        assert!(profile.report().is_null());
        profile.add(&update);
        profile.add(&update);
        let report = profile.report();
        assert_eq!(report["updates"], 2);
        assert_eq!(report["wall_seconds"], 200.0);
        assert_eq!(report["stages"][8]["seconds"], 24.0);
        assert_eq!(report["stages"][6]["fraction"], 0.6);
        assert_eq!(report["r1_step_wait_over_half"], true);
        let sum = report["stages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["fraction"].as_f64().unwrap())
            .sum::<f64>();
        assert!((sum - 1.0).abs() < 1e-12);
        let mut csv = Vec::new();
        Update::write_header(&mut csv).unwrap();
        update.write(&mut csv, 1).unwrap();
        let csv = String::from_utf8(csv).unwrap();
        assert_eq!(csv.lines().count(), 2);
        assert!(csv.lines().all(|line| line.split(',').count() == 11));
    }
}
