"""Feast: point-in-time offline join и материализация в online SQLite."""

from datetime import timedelta
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from feast import FeatureStore

from data import EVENT_TIME, FEATURES, ID, ROOT, SEED
from feast_repo.features import VIEW, definitions

REFS = [f'{VIEW}:{name}' for name in FEATURES]


def build_store(work, frame):
    repo = Path(work).resolve() / 'feast'
    repo.mkdir(parents=True, exist_ok=True)
    source = repo / 'offline.parquet'
    frame.to_parquet(source, index=False)
    copyfile(ROOT / 'feast_repo' / 'feature_store.yaml', repo / 'feature_store.yaml')
    store = FeatureStore(repo_path=str(repo))
    store.apply(definitions(source.as_posix()))
    events = frame[[ID]].assign(event_timestamp=EVENT_TIME + timedelta(hours=1))
    historical = store.get_historical_features(entity_df=events, features=REFS).to_df().set_index(ID)
    offline = historical.loc[frame[ID], list(FEATURES)].astype(float)
    store.materialize(EVENT_TIME.to_pydatetime() - timedelta(hours=1),
                      EVENT_TIME.to_pydatetime() + timedelta(hours=2))
    sample = frame.sample(n=min(256, len(frame)), random_state=SEED).set_index(ID)[list(FEATURES)]
    result = online_features(store, sample.index.tolist())
    np.testing.assert_allclose(offline.loc[sample.index], sample, rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(result, sample, rtol=0, atol=1e-9, equal_nan=True)
    deltas = np.abs(result.to_numpy() - sample.to_numpy())
    return offline, {'checked': len(sample), 'max_abs_diff': float(np.nanmax(deltas)), 'passed': True}


def online_features(store, ids):
    values = store.get_online_features(features=REFS, entity_rows=[{ID: int(key)} for key in ids]).to_dict()
    frame = pd.DataFrame(values).set_index(ID)[list(FEATURES)].astype(float)
    if frame.isna().all(axis=1).any():
        raise KeyError('Запись в online store отсутствует')
    return frame
