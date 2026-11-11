"""FastAPI: версия из Model Registry, online Feast и объяснение прогноза."""

import argparse
import hashlib
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from pydantic import BaseModel, Field

from data import FEATURES
from store import online_features
from training import MODEL, configure_tracking, get_version, shap_linear

LOG = logging.getLogger(__name__)


class Request(BaseModel):
    patient_id: int = Field(gt=0, le=2**63 - 1, strict=True)
    request_id: str | None = Field(default=None, min_length=1, max_length=100)
    explain: bool = False


def in_canary(request_id):
    bucket = hashlib.sha256(request_id.encode()).digest()
    return int.from_bytes(bucket[:4], 'big') % 100 < 5


def predict(version, frame):
    model, number, meta = version
    probability = float(model.predict_proba(frame)[0, 1])
    return {'probability': probability, 'alert': probability >= meta['threshold'], 'version': number}


class Rollout:
    def __init__(self, client, mode):
        self.client = client
        self.mode = mode
        self.champion = get_version(client, 'champion')
        self.candidate = get_version(client, 'candidate')

    def finish(self, approved):
        self.mode = 'stable'
        if approved:
            self.client.set_registered_model_alias(MODEL, 'champion', self.candidate[1])
            self.champion = self.candidate
        return self.champion[1]


def create_app(work, mode='stable'):
    client = configure_tracking(work)
    release = Rollout(client, mode)
    store = FeatureStore(repo_path=str(Path(work).resolve() / 'feast'))
    app = FastAPI(title='Hospital Readmissions')
    app.state.release = release

    @app.get('/health')
    async def health():
        return {'status': 'ready', 'version': release.champion[1], 'mode': release.mode}

    @app.post('/predict')
    async def infer(request: Request):
        try:
            frame = online_features(store, [request.patient_id])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='Признаки не найдены') from exc
        except Exception as exc:
            LOG.exception('Feast online недоступен')
            raise HTTPException(status_code=503, detail='Хранилище признаков недоступно') from exc
        selected = release.champion
        if release.mode == 'canary' and in_canary(request.request_id or str(request.patient_id)):
            selected = release.candidate
        result = predict(selected, frame)
        if release.mode == 'shadow':
            shadow = predict(release.candidate, frame)
            LOG.info('shadow_delta=%+.6f', shadow['probability'] - result['probability'])
        if request.explain:
            model, _, meta = selected
            base, values = shap_linear(model, frame, meta['baseline'])
            result['explanation'] = {'base_log_odds': base, 'log_odds': float(base + values[0].sum()),
                                     'shap': dict(zip(FEATURES, map(float, values[0]), strict=True))}
        return result

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', default='tmp/local')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--mode', choices=['stable', 'canary', 'shadow'], default='stable')
    args = parser.parse_args()
    uvicorn.run(create_app(args.work, args.mode), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
