"""Record combined offline validation with its actual process exit code."""
import pathlib
import subprocess
import sys

root=pathlib.Path(__file__).resolve().parents[1]
log=root/'validation'/'current_run.log'
with log.open('w',encoding='utf-8') as stream:
    result=subprocess.run([sys.executable,str(root/'validation'/'run.py'),
        '--reuse-oracles','--upstream-python'],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
print('\n'.join(log.read_text(encoding='utf-8').splitlines()[-8:]))
print(f'Combined validation exit code: {result.returncode}; log: {log}')
raise SystemExit(result.returncode)
