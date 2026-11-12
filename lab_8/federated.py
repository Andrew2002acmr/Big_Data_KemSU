"""Локальная симуляция трёх клиентов Flower с взвешенной агрегацией FedAvg."""

from importlib.metadata import version

import numpy as np
from scipy.special import expit

from evaluation import features, score
from mock_insurance import TARGET


def simulate(train, test):
    from flwr.client import NumPyClient
    from flwr.server.strategy.aggregate import aggregate

    class Insurer(NumPyClient):
        def __init__(self, local_data, client_id):
            self.x = features(local_data)
            self.y = local_data[TARGET].to_numpy()
            self.rng = np.random.default_rng(client_id)

        def fit(self, parameters, config):
            # Вход и ответ содержат только параметры и размер локальной выборки.
            weights, intercept = [item.copy() for item in parameters]
            for _ in range(config['local_epochs']):
                order = self.rng.permutation(len(self.x))
                for start in range(0, len(order), 256):
                    ix = order[start:start + 256]
                    residual = expit(self.x[ix] @ weights + intercept[0]) - self.y[ix]
                    weights -= .34 * (self.x[ix].T @ residual / len(ix))
                    intercept -= .34 * residual.mean()
            return [weights, intercept], len(self.x), {}

    groups = [('Центр',), ('Сибирь',), ('Урал', 'Юг')]
    clients = [Insurer(train[train.district.isin(group)], index + 31) for index, group in enumerate(groups)]
    assert sum(len(client.x) for client in clients) == len(train)
    x = features(test)
    params = [np.zeros(x.shape[1]), np.zeros(1)]
    history = []
    for round_number in range(1, 13):
        updates = [client.fit(params, {'local_epochs': 2}) for client in clients]
        params = aggregate([(params, count) for params, count, _ in updates])
        history.append({'round': round_number, **score(test, expit(x @ params[0] + params[1][0]))})
    return {'flower_version': version('flwr'), 'clients': [len(client.x) for client in clients], 'rounds': history,
            'final': history[-1], 'message': 'параметры + число локальных примеров',
            'network': 'in-process', 'formal_dp': False}
