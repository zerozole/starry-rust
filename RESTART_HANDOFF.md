# Current handoff

Project: D:\Games\starry-rust. Reference: ../starry-upstream, commit
b72dff08588532f96bd072f2f1005e227d8e4ed8. New Python API accepted.

Latest work completes System surface/RV Jacobians, source-normalized Doppler
solvers and Jacobians, boundary audits, degree32 stress checks, independent
references and source orbital conventions. Documentation has been consolidated.
Final release verification passed: Windows integrated exit 0; Linux 47 Rust/77
Python tests; rebuilt Windows wheel and every numerical-guide example passed.

Current requirements and limits: COMPLETION_CHECKLIST.md, NUMERICAL_LIMITS.md,
NUMERICAL_API.md and VALIDATION.md. Historical handoffs are archived under
validation/history/pre_completion_20260916/; do not revive superseded gaps.

## Reproduce

Windows: python validation/record_run.py (captures the true subprocess exit).
Wheel: python setup.py bdist_wheel; install into .wheel-smoke with pip
--no-index --no-deps --upgrade --target, then python -I validation/wheel_smoke.py.
Linux: validation/linux_python.sh; Rust tests use --target-dir target-linux.
Do not share target/ between Windows and Linux.

Source-style orbital geometry/RV is selected explicitly with
orbit_convention='starry'. Barycentric remains the backward-compatible default.
Source-style Doppler normalization requires continuum_index; the integrated
continuum default is retained. Neither option uses reference code at runtime.

No work on EB, scope-ml, supernova fitting, MESA or remote cluster state is
part of this task. Central Codex memories were read but not edited.
