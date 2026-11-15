use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map_or_else(
            || "unknown".into(),
            |output| String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        )
}

pub(crate) fn total_memory() -> u64 {
    sysinfo::System::new().map_or(0, |system| system.total_memory())
}

pub(crate) fn current_process_memory() -> u64 {
    let Ok(pid) = sysinfo::get_current_pid() else {
        return 0;
    };
    let Ok(mut system) = sysinfo::System::new() else {
        return 0;
    };
    system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    system.process(pid).map_or(0, sysinfo::Process::memory)
}

pub(crate) fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}
