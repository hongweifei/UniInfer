use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::measure::Measurement;

pub(crate) fn compare(
    baseline_path: &Path,
    current_path: &Path,
    warn: f64,
    fail: f64,
) -> Result<()> {
    let baseline: Measurement = serde_json::from_str(
        &std::fs::read_to_string(baseline_path)
            .with_context(|| format!("read {}", baseline_path.display()))?,
    )?;
    let current: Measurement = serde_json::from_str(
        &std::fs::read_to_string(current_path)
            .with_context(|| format!("read {}", current_path.display()))?,
    )?;

    for (name, baseline_value, current_value) in [
        (
            "schema",
            baseline.schema.to_string(),
            current.schema.to_string(),
        ),
        ("model", baseline.model.clone(), current.model.clone()),
        ("engine", baseline.engine.clone(), current.engine.clone()),
        ("device", baseline.device.clone(), current.device.clone()),
        ("ep", baseline.ep.clone(), current.ep.clone()),
        ("commit", baseline.commit.clone(), current.commit.clone()),
    ] {
        if baseline_value != current_value {
            bail!(
                "baseline mismatch for {name}: baseline {baseline_value}, current {current_value}"
            );
        }
    }

    let checks = [
        (
            "ttft_ms",
            ratio(current.ttft_ms, baseline.ttft_ms),
            true,
            "ms",
        ),
        (
            "total_ms",
            ratio(current.total_ms, baseline.total_ms),
            true,
            "ms",
        ),
        (
            "tokens_per_second",
            ratio(baseline.tokens_per_second, current.tokens_per_second),
            false,
            "tok/s",
        ),
        (
            "peak_rss_bytes",
            ratio(
                current.peak_rss_bytes as f64,
                baseline.peak_rss_bytes as f64,
            ),
            true,
            "bytes",
        ),
    ];

    let mut failed = false;
    for (name, degradation, lower_is_better, unit) in checks {
        if !degradation.is_finite() {
            continue;
        }
        let change = degradation * 100.0;
        let direction = if lower_is_better { "higher" } else { "lower" };
        if degradation > fail {
            println!("FAIL {name}: {change:.1}% {direction} is worse ({unit})");
            failed = true;
        } else if degradation > warn {
            println!("WARN {name}: {change:.1}% {direction} is worse ({unit})");
        } else {
            println!("ok   {name}: {change:+.1}%");
        }
    }

    if failed {
        bail!("benchmark regression exceeds {:.0}%", fail * 100.0);
    }
    Ok(())
}

pub(crate) fn ratio(current: f64, baseline: f64) -> f64 {
    if baseline <= 0.0 {
        f64::NAN
    } else {
        current / baseline - 1.0
    }
}
