"""Build platform wheels containing the dependency-free Rust shared library."""
import pathlib
import shutil
import subprocess
import sys
from setuptools import setup, Distribution
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

ROOT = pathlib.Path(__file__).resolve().parent


class Build(build_py):
    def run(self):
        subprocess.run(['cargo', 'build', '--release', '--offline'], cwd=ROOT, check=True)
        super().run()
        name = 'starry_rust.dll' if sys.platform == 'win32' else (
            'libstarry_rust.dylib' if sys.platform == 'darwin' else 'libstarry_rust.so')
        destination = pathlib.Path(self.build_lib) / '_starry_native' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'target' / 'release' / name, destination)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class Wheel(bdist_wheel):
    def get_tag(self):
        _, _, platform = super().get_tag()
        return 'py3', 'none', platform


setup(cmdclass={'build_py': Build, 'bdist_wheel': Wheel}, distclass=BinaryDistribution)
