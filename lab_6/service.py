"""ONNX HTTP-сервис и потребитель Kafka, масштабируемый KEDA."""

import json
import os
import socket
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from time import perf_counter, sleep

import numpy as np
from confluent_kafka import Consumer
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from common import FEATURES, dumps, redis_client, scores, session

VERSION = os.getenv('RELEASE_VERSION', 'v1')
MODEL = session()
BMI = deque(maxlen=512)
LOCK = Lock()
BINS = np.array([-np.inf, 22, 26, 30, 34, 38, np.inf])
EXPECTED = np.histogram(np.linspace(18, 40, 512), BINS)[0] / 512
REQUESTS = Counter('lab6_http_requests', 'Inference requests', ['code'])
LATENCY = Histogram('lab6_inference_seconds', 'Inference time', buckets=(.001, .005, .01, .02, .05, .1, .25, .5, 1))
DRIFT = Gauge('lab6_data_drift_psi', 'BMI drift PSI')
CONSUMED = Counter('lab6_queue_processed', 'Completed Kafka events')


def update_drift(rows):
    with LOCK:
        BMI.extend(float(row[1]) for row in rows)
        if len(BMI) >= 64:
            a = np.maximum(EXPECTED, 1e-5)
            b = np.maximum(np.histogram(BMI, BINS)[0] / len(BMI), 1e-5)
            DRIFT.set(float(np.sum((b - a) * np.log(b / a))))


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def setup(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        super().setup()

    def write(self, status, data, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/health':
            self.write(200, dumps({'status': 'ready', 'version': VERSION}).encode())
        elif self.path == '/metrics':
            self.write(200, generate_latest(), CONTENT_TYPE_LATEST)
        else:
            self.write(404, b'{}')

    def do_POST(self):
        if self.path != '/score':
            self.write(404, b'{}')
            return
        start = perf_counter()
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 64_000:
                raise ValueError('Invalid request size')
            rows = json.loads(self.rfile.read(size))['features']
            matrix = np.asarray(rows, dtype=np.float32)
            if matrix.ndim != 2 or not 0 < len(matrix) <= 256 or matrix.shape[1] != len(FEATURES) or not np.isfinite(matrix).all():
                raise ValueError('Invalid feature shape')
            if os.getenv('BAD_RELEASE') == '1':
                status, response = 503, {'error': 'simulated candidate failure', 'version': VERSION}
            else:
                status, response = 200, {'version': VERSION, 'scores': scores(MODEL, matrix).tolist()}
                update_drift(matrix)
        except (TypeError, ValueError, KeyError):
            status, response = 422, {'error': 'invalid request'}
        REQUESTS.labels(str(status)).inc()
        LATENCY.observe(perf_counter() - start)
        self.write(status, dumps(response).encode())

    def log_message(self, _format, *args):
        pass


def consume():
    kafka = Consumer({'bootstrap.servers': 'kafka:9092', 'group.id': 'lab6-model', 'auto.offset.reset': 'earliest',
                      'enable.auto.commit': False})
    db = redis_client()
    kafka.subscribe(['scale-events'])
    try:
        while True:
            message = kafka.poll(1)
            if message is None or message.error():
                continue
            event = json.loads(message.value())
            score = float(scores(MODEL, [event['features']])[0])
            sleep(.065)
            db.set(f"lab6:queued:{event['id']}", score, nx=True)
            kafka.commit(message=message, asynchronous=False)
            CONSUMED.inc()
    finally:
        kafka.close()


if __name__ == '__main__':
    if os.getenv('KAFKA_WORKER') == '1':
        Thread(target=consume, daemon=True).start()
    ThreadingHTTPServer(('0.0.0.0', 8000), Handler).serve_forever()
