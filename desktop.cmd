@echo off
setlocal
cd /d "%~dp0"
cargo run --manifest-path apps\desktop\Cargo.toml %*
