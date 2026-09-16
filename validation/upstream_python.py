"""Run explicitly selected, unmodified upstream tests against the eager adapter."""
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / '.test-deps')]
import starry_compat
sys.modules['starry'] = starry_compat
import pytest


class Report:
    def __init__(self):
        self.cases = {}
    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.failed:
            self.cases[report.nodeid] = {
                'outcome': report.outcome,
                'message': str(report.longrepr) if report.failed else None,
            }


if __name__ == '__main__':
    files = sys.argv[1:] or ['test_intensity.py', 'test_rotate_greedy.py',
        'test_indices_greedy.py', 'test_reflected_normalization.py',
        'test_solve_greedy.py', 'test_system_solve_greedy.py']
    if not sys.argv[1:]:
        files += ['test_ld.py', 'test_oblate_system.py', 'test_oren_nayar.py', 'test_multi_wavelength_greedy.py']
        files += ['test_minimize.py::test_sturm', 'test_minimize.py::test_limbdark_physical', 'test_minimize.py::test_bounded_minimize']
    paths = [str(ROOT.parent / 'starry-upstream' / 'tests' / 'greedy' / f) for f in files]
    report = Report()
    started = time.monotonic()
    status = pytest.main(['-q', '--tb=short', '-p', 'no:cacheprovider', *paths], plugins=[report])
    payload = {'files': files, 'exit_code': int(status), 'elapsed_seconds': time.monotonic()-started,
               'cases': report.cases, 'passed': int(status)==0}
    (ROOT/'validation'/'upstream_python_report.json').write_text(json.dumps(payload, indent=2)+'\n')
    raise SystemExit(status)
