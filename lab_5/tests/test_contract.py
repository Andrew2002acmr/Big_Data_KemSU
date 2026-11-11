import hashlib
import json
import unittest

import numpy as np

from benchmark import sensor_psi
from common import FEATURES, MODEL_PATH, build_features, scores, session


class ContractTest(unittest.TestCase):
    def test_feature_order_and_model(self):
        metadata = json.loads(MODEL_PATH.with_suffix('.json').read_text(encoding='utf-8'))
        self.assertEqual(tuple(metadata['features']), FEATURES)
        self.assertEqual(metadata['onnx_sha256'], hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest())
        row = build_features([50., 25., 0., 1.], {'stay': 3, 'destination': 'Rehab'})
        self.assertEqual(row, [50., 25., 0., 1., 3, 1., 0.])
        prediction = scores(session(), [row, row])
        np.testing.assert_allclose(prediction[0], prediction[1], rtol=0, atol=1e-7)
        self.assertTrue(0 < prediction[0] < 1)

    def test_psi_detects_shift_outside_reference_bins(self):
        data = np.linspace(20, 40, 100)
        self.assertEqual(sensor_psi(data, data), 0)
        self.assertGreater(sensor_psi(data, data + 12), .2)
