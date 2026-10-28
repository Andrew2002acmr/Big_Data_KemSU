"""Download only the two public source datasets used in the report."""
from pathlib import Path
import urllib.request
import zipfile

root=Path(__file__).parent/'tmp'/'data'
root.mkdir(parents=True,exist_ok=True)
sources={
    'hospital_kaggle.zip': 'https://api.kaggle.com/v1/datasets/download/siddharth0935/hospital-readmission-predictionsynthetic-dataset',
    'yellow_tripdata_2024-01.parquet': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet',
}
for name,url in sources.items():
    path=root/name
    if not path.exists():
        with urllib.request.urlopen(url,timeout=60) as response, path.open('wb') as out:
            while block:=response.read(1024*1024): out.write(block)
    print(name,path.stat().st_size)
with zipfile.ZipFile(root/'hospital_kaggle.zip') as archive:
    for name in archive.namelist():
        if name.endswith(('.csv','.json')):
            (root/Path(name).name).write_bytes(archive.read(name))
