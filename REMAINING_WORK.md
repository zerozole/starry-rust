# Remaining work

The previously identified numerical-port completion tasks are closed as of
2026-09-16. See COMPLETION_CHECKLIST.md for item-by-item evidence and
VALIDATION.md for test results. The supported profile is NUMERICAL_LIMITS.md.

## Boundaries and optional follow-up

- macOS native execution and wheel packaging are untested.
- No benchmarked speedup over upstream or real-data scientific validation is claimed.
- Arbitrary symbolic graphs, plotting widgets, generic flux Hessians and new
  joint nonlinear posterior models are outside the agreed numerical API scope.
- Map degree32 and combined-product/filter limits are explicit; higher-degree
  extensions require additional conditioning and convergence work.
- Source and default orbital/continuum conventions differ intentionally. Use
  the explicit source options in NUMERICAL_API.md when reproducing upstream.

Newly discovered discrepancies should be added here with a reproducer. Passing
finite reference suites does not establish correctness for every valid input.
The earlier open-work list is preserved under
validation/history/pre_completion_20260916/REMAINING_WORK.md.
