import sys, os, subprocess
import glob
import setuptools

CURRENT_DIR = os.path.dirname(__file__)

def get_version():
    """Get library version"""
    # We shall not import version.py in setup.py directly since __init__.py
    # will be invoked which introduces dependences
    version_py = os.path.join(CURRENT_DIR, './dbinfer_bench/version.py')
    env = {'__file__' : version_py}
    exec(compile(open(version_py, 'rb').read(), version_py, 'exec'), env, env)
    version = env['__version__']
    return version

VERSION = get_version()

# Install
setuptools.setup(
    name='dbinfer-bench',
    version=VERSION,
    packages=['dbinfer_bench'],
    author='AWS Shanghai AI Lab',
    python_requires='>=3.7',
    py_modules=['dbinfer_bench'],
    include_package_data=True,
    package_data={
        '': ['download_config.yaml']
    },
    install_requires=[
        'pydantic',
        'numpy',
        'pandas',
        'boto3',
        'requests[security]',
        'pyyaml',
        'dgl>=2.1a240205',
        'sqlalchemy',
    ],
    license='APACHE',
    url='https://github.com/awslabs/multi-table-benchmark',
)
