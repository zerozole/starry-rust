#!/bin/sh
set -eu
cd /mnt/d/Games/starry-rust
/home/atal/.cargo/bin/cargo build --offline --release --target-dir target-linux
export STARRY_RUST_LIBRARY=/mnt/d/Games/starry-rust/target-linux/release/libstarry_rust.so
export PYTHONDONTWRITEBYTECODE=1
python_bin=${STARRY_TEST_PYTHON:-/mnt/d/Games/eb-physical-fitter/.venv/bin/python}
"$python_bin" -m unittest discover -s python -v > validation/linux_python.log 2>&1
tail -n 5 validation/linux_python.log
