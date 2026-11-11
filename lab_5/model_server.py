"""Локальный HTTP-сервис пакетного ONNX-инференса."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from common import FEATURES, MODEL_VERSION, scores, session


@asynccontextmanager
async def lifespan(app):
    app.state.model = session()
    yield


app = FastAPI(title='Hospital readmissions ONNX', lifespan=lifespan)


class Batch(BaseModel):
    features: list[list[float]] = Field(min_length=1, max_length=2048)


@app.get('/health')
def health():
    return {'model': MODEL_VERSION}


@app.post('/score')
def score(request: Batch):
    if any(len(row) != len(FEATURES) for row in request.features):
        raise HTTPException(422, detail='Invalid feature vector length')
    return {'model': MODEL_VERSION, 'scores': scores(app.state.model, request.features).tolist()}
