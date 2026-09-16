"""Compare Rust to unmodified upstream C++ kernels at a pinned commit.

Requires the sibling starry-upstream checkout, g++, and a release Rust build.
No installed Python starry package or Python scientific dependencies needed.
"""
import json
import hashlib
import math
import pathlib
import random
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "starry-upstream"
PIN = "b72dff08588532f96bd072f2f1005e227d8e4ed8"
SUFFIX = ".exe" if sys.platform == "win32" else ""
ORACLE = ROOT / "validation" / ("oracle" + SUFFIX)
RUST = ROOT / "target" / "release" / ("probe" + SUFFIX)


def values(exe, *args):
    return [float(v) for v in subprocess.check_output([str(exe), *map(str, args)], text=True).split()]


def main():
    git = ["git", "-c", f"safe.directory={UPSTREAM.as_posix()}", "-C", str(UPSTREAM)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if head != PIN:
        raise RuntimeError(f"upstream pin changed: {head}")
    dirty = subprocess.check_output([*git, "status", "--porcelain"], text=True)
    if dirty.strip():
        raise RuntimeError("upstream checkout must be unmodified")
    results = {}
    count = 0

    def compare(category, args, atol, rtol=0.0, transpose=False):
        nonlocal count
        upstream_args = ["flux", *args[1:]] if args[0] == "analytic" else args
        a = values(ORACLE, *upstream_args)
        b = values(RUST, *args)
        if transpose:
            n = math.isqrt(len(a))
            a = [a[j*n+i] for i in range(n) for j in range(n)]
        if len(a) != len(b):
            raise AssertionError("shape mismatch")
        errors = [abs(x-y) for x, y in zip(a, b)]
        good = all(math.isfinite(x) and math.isfinite(y) and abs(x-y) <= atol + rtol*abs(x) for x, y in zip(a, b))
        record = results.setdefault(category, {"cases": 0, "values": 0, "max_absolute_error": 0.0, "failures": []})
        record["cases"] += 1
        record["values"] += len(a)
        record["max_absolute_error"] = max(record["max_absolute_error"], max(errors))
        if not good:
            record["failures"].append({"args": args, "max_error": max(errors)})
            print("FAIL", args, max(errors), flush=True)
        count += 1

    start = time.monotonic()
    for degree in range(13):
        compare("basis", ["basis", degree], 2e-10, 2e-13)
    print("Basis comparisons complete", flush=True)
    for degree in [0, 1, 2, 5, 10]:
        for b, r in [(0., .1), (0., .9), (.5, .5), (.3, .3), (.8, .8), (1., .1),
                     (.5, .1), (.9, .1), (.4, .6), (.6, .4), (1.1, .1),
                     (2., .1), (0., 2.), (.7, 1.2), (1.49, .5), (.01, 1.),
                     (1e-6, .1), (.10000001, .1), (.4, 1.3999999)]:
            compare("flux", ["flux", degree, b, r], 2e-8)
            compare("analytic", ["analytic", degree, b, r], 2e-8)
    rng = random.Random(721)
    for _ in range(40):
        degree = rng.choice([1, 2, 5, 8])
        r = 10 ** rng.uniform(-2, .3)
        b = rng.uniform(0., 1.+r)
        compare("flux_random", ["flux", degree, b, r], 2e-8)
        compare("analytic_random", ["analytic", degree, b, r], 2e-8)
    print("Flux comparisons complete", flush=True)
    for degree in [0, 1, 3, 5]:
        for b, r in [(0., .1), (.2, .2), (.7, .4), (1.2, .5), (2., .1)]:
            compare("limb", ["limb", degree, b, r], 2e-8)
    for degree in [0, 1, 2, 5, 10]:
        for b, r in [(.3, .1), (.2, .2), (.7, .4), (.3, .8), (1.2, .5), (.2, 1.1), (.9, .15)]:
            compare("gradient", ["gradient", degree, b, r], 2e-8)
    # Harness follows maps.py: dotR(I, -theta). Transpose to column coefficients.
    for degree in [1, 2, 5]:
        for axis in [(1., 0., 0.), (0., 0., 1.), (1/3, 2/3, 2/3)]:
            for theta in [0., .73, math.pi]:
                compare("rotation", ["rotation", degree, *axis, theta], 2e-10, transpose=True)
    for k in [0., 1e-12, .01, .5, .99, 1.-1e-12]:
        for p, a, b in [(1., 1., 0.), (1., 1., 1.-k), (.3, 0., 2.), (-.3, .7, -.2)]:
            compare("cel", ["cel", 0, k, p, a, b], 1e-12, 2e-14)
    files = [*sorted((ROOT / "src").rglob("*.rs")), *sorted((ROOT / "tests").rglob("*.rs")),
             *sorted((ROOT / "python").glob("*.py")), ROOT / "Cargo.toml", ROOT / "Cargo.lock",
             ROOT / "validation" / "oracle.cpp", pathlib.Path(__file__).resolve(), ORACLE, RUST]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = {"upstream_commit": PIN, "source_and_binary_sha256": hashes, "elapsed_seconds": time.monotonic()-start,
              "cases": count, "results": results,
              "passed": all(not r["failures"] for r in results.values())}
    (ROOT / "validation" / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
