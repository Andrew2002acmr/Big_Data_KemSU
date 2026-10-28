"""Run experiments sequentially so engines do not compete for the same machine."""
from pathlib import Path
import subprocess,sys

ROOT=Path(__file__).parent


def call(script,*args):
    subprocess.run([sys.executable,str(ROOT/script),*args],cwd=ROOT,check=True)


if __name__=='__main__':
    phase=sys.argv[1] if len(sys.argv)>1 else 'all'
    if phase in ['all','prepare']:call('prepare.py')
    if phase in ['all','bench']:
        for engine in ['pandas','dask','spark','polars','duckdb']:
            for size in ['full','100mb','1gb']:call('benchmark.py',engine,size,'3')
    if phase in ['all','opt']:call('spark_optimize.py','1gb','3')
    if phase in ['all','report']:call('report.py')
