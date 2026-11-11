"""Общее определение версии признаков для обучения и инференса."""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64

from data import FEATURES, ID

VIEW = 'discharge_v1'


def definitions(path):
    patient = Entity(name='patient', join_keys=[ID], value_type=ValueType.INT64)
    source = FileSource(path=path, timestamp_field='event_timestamp')
    view = FeatureView(name=VIEW, entities=[patient], ttl=timedelta(days=1),
                       schema=[Field(name=name, dtype=Float64) for name in FEATURES],
                       source=source, online=True)
    return [patient, view]
