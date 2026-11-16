#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
exec cargo run --manifest-path apps/desktop/Cargo.toml "$@"
