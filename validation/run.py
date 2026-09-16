"""Offline build + validation entry point. No downloads or source modifications."""
import os
import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(*args):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-python',action='store_true',help='also run selected unmodified upstream greedy tests; requires pytest')
    parser.add_argument('--reuse-oracles',action='store_true',help='reuse previously built C++ reference executables')
    args=parser.parse_args()
    run("cargo", "fmt", "--all", "--", "--check")
    run("cargo", "clippy", "--offline", "--all-targets", "--", "-D", "warnings")
    run("cargo", "test", "--offline")
    run("cargo", "build", "--release", "--offline")
    for name in ['oracle','reflected_oracle','oblate_oracle']:
        oracle = f"validation/{name}.exe" if sys.platform == "win32" else f"validation/{name}"
        if args.reuse_oracles:
            if not (ROOT/oracle).is_file():raise FileNotFoundError(oracle)
            continue
        run(os.environ.get("CXX", "g++"), "-std=c++14", "-O1", "-DSTARRY_BRANCHING_DISABLE_OPTIM",
            "-I../starry-upstream/starry/_core/ops/lib/include",
            "-I../starry-upstream/starry/_core/ops/lib/vendor/eigen_3.3.5",
            f"validation/{name}.cpp", "-o", oracle)
    run(sys.executable, "validation/compare.py")
    run(sys.executable, "validation/compare_reflected.py")
    run(sys.executable, "validation/compare_oblate.py")
    run(sys.executable, "validation/compare_specialized_gradients.py")
    run(sys.executable, "validation/compare_oblate_derivatives.py")
    run(sys.executable, "validation/compare_rv_derivatives.py")
    run(sys.executable, "validation/independent_system.py")
    run(sys.executable, "validation/independent_velocity.py")
    run(sys.executable, "validation/compare_source_solvers.py")
    run(sys.executable, "validation/compare_source_orbits.py")
    run(sys.executable, "validation/compare_specialized_high.py")
    run(sys.executable, "validation/compare_rings.py")
    run(sys.executable, "validation/decimal_rings.py")
    run(sys.executable, "validation/feature_audit.py")
    run(sys.executable, "validation/source_boundaries.py")
    run(sys.executable, "-m", "unittest", "discover", "-s", "python", "-v")
    if args.upstream_python:
        run(sys.executable,'validation/upstream_python.py')
    result = subprocess.run(["cargo", "run", "--release", "--offline", "--example", "transit"],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    rows = result.stdout.strip().splitlines()
    if len(rows) != 102 or rows[0] != "x,flux":
        raise AssertionError("unexpected transit example output")
    (ROOT / "validation" / "transit.csv").write_text(result.stdout, encoding="utf-8")
    print("All checks passed. Reports: validation/report.json and validation/transit.csv")


if __name__ == "__main__":
    main()
