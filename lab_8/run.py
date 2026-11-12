"""Локальный запуск шести частей. Данные и обученные модели хранятся в tmp."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dp_experiment import EPSILONS, WORK, dp_sgd, sweep
from evaluation import dcr_audit, utility
from federated import simulate
from mock_insurance import SEED, TARGET, partition, profile, relational, validate_relations
from privacy import k_audit, pii_audit
from release_audit import RELEASE_EPSILON, gates
from synthesis import synthesis


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda item: item.item() if isinstance(item, np.generic) else str(item)) + '\n',
                    encoding='utf-8', newline='\n')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def graphics(output, anonymity, dp):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.3, 4.6), layout='constrained')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    choices = [anonymity['baseline'], *anonymity['choices']]
    positions = range(len(choices))
    losses = [100 * item['information_loss'] for item in choices]
    labels = ['Исходные', 'k ≥ 2', 'k ≥ 5', 'k ≥ 10']
    ax.plot(positions, losses, marker='o', color='#245e94', linewidth=2)
    ax.set_xticks(list(positions), labels)
    for i, (loss, item) in enumerate(zip(losses, choices, strict=True)):
        ax.annotate(f'k_min = {item["k_min"]}', (i, loss), xytext=(0, 8), textcoords='offset points', ha='center')
    ax.set(ylabel='Потеря информации NCP, %', title='Обобщение квазиидентификаторов', ylim=(0, max(losses) + 8))
    ax.grid(axis='y', alpha=.25)
    fig.savefig(output / 'anonymity.png', dpi=155)

    fig = Figure(figsize=(7.3, 4.6), layout='constrained')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for name, label, color in [('opendp', 'OpenDP', '#245e94'), ('mwem', 'SmartNoise MWEM', '#b76439'),
                               ('mostlyai', 'MOSTLY AI', '#43774d')]:
        values = [item for item in dp['runs'] if item['method'] == name]
        if values:
            ax.plot([row['requested_epsilon'] for row in values], [100 * row['relative_auc'] for row in values],
                    label=label, color=color, marker='o')
    ax.axhline(90, color='#6b7280', linestyle='--', label='Гейт 90 %')
    ax.set(xlabel='Заданный бюджет ε', ylabel='ROC-AUC TSTR от baseline, %', title='Полезность DP-синтеза')
    ax.set_xticks(EPSILONS)
    ax.grid(axis='y', alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(output / 'dp_utility.png', dpi=155)


class Experiment:
    def __init__(self, reuse=False):
        WORK.mkdir(exist_ok=True)
        self.reuse = reuse
        self.original = profile()
        self.raw = relational(self.original, 'raw', SEED, include_pii=True)
        self.train, self.test, self.control = partition(self.original)
        self.baseline = utility(self.train, self.test)[0]

    def pii(self):
        path = WORK / 'privacy.json'
        return read_json(path)['pii'] if self.reuse and path.is_file() else pii_audit(self.raw['clients'])

    def anonymity(self, pii):
        path = WORK / 'privacy.json'
        if self.reuse and path.is_file():
            return read_json(path)['k']
        summary = k_audit(self.original)
        write_json(path, {'pii': pii, 'k': summary})
        return summary

    def synthesize(self):
        path = WORK / 'sdv.json'
        names = ('gaussian_copula', 'ctgan')
        if self.reuse and path.is_file() and all((WORK / f'{name}.parquet').is_file() for name in names):
            return read_json(path), {name: pd.read_parquet(WORK / f'{name}.parquet') for name in names}
        summary, samples = synthesis(self.train, self.test, self.raw)
        for name, sample in samples.items():
            sample.to_parquet(WORK / f'{name}.parquet', index=False)
        write_json(path, summary)
        return summary, samples

    def differentially_private(self):
        open_path, sgd_path, comparison_path = (WORK / name for name in ('open.json', 'sgd.json', 'dp_comparison.json'))
        release_path = WORK / f'opendp-{RELEASE_EPSILON}.parquet'
        baseline = self.baseline['auc']
        if self.reuse and open_path.is_file() and release_path.is_file():
            opened, release = read_json(open_path), pd.read_parquet(release_path)
        else:
            opened, samples = sweep(self.train, self.test, baseline, methods=('opendp',))
            release = samples[('opendp', RELEASE_EPSILON)]
            release.to_parquet(release_path, index=False)
            write_json(open_path, opened)
        if self.reuse and sgd_path.is_file():
            sgd = read_json(sgd_path)
        else:
            sgd = dp_sgd(self.train, self.test)
            write_json(sgd_path, sgd)
        if self.reuse and comparison_path.is_file():
            external = read_json(comparison_path)
        else:
            external, _ = sweep(self.train, self.test, baseline, methods=('mwem', 'mostlyai'),
                                include_low_volume=False, reuse=self.reuse)
            write_json(comparison_path, external)
        budget = [*opened['ledger'], *external['ledger'], sgd['privacy']]
        open_budget = [row['epsilon_upper'] for row in opened['ledger']]
        return {'runs': [*opened['runs'], *external['runs']], 'low_volume': opened['low_volume'],
                'dp_sgd': sgd, 'ledger': budget,
                'opendp_joint_budget': {'epsilon': sum(open_budget), 'delta': 0}}, release

    def evaluate(self, dp, sample, non_private):
        path = WORK / 'release.json'
        if self.reuse and path.is_file():
            return read_json(path)
        budget = [row for row in dp['ledger'] if row['method'] == 'OpenDP'
                  and row['requested_epsilon'] == RELEASE_EPSILON and row['operation'].startswith('main:')]
        audit = gates(self.train, self.test, self.control, sample, self.raw, self.baseline['auc'], budget)
        audit['non_private_distances'] = {name: dcr_audit(self.train, self.control, table)
                                          for name, table in non_private.items()}
        write_json(path, audit)
        return audit

    def finish(self, pii, anonymity, synthetic, dp, audit, output):
        output.mkdir(parents=True, exist_ok=True)
        federated = simulate(self.train, self.test)
        report = {'variant': 6, 'seed': SEED, 'mock_tables': validate_relations(self.raw),
                  'split': {name: {'rows': len(frame), 'claims': int(frame[TARGET].sum())}
                            for name, frame in [('train', self.train), ('test', self.test), ('control', self.control)]},
                  'pii': pii, 'k': anonymity, 'baseline': self.baseline, 'synthesis': synthetic, 'dp': dp,
                  'release': audit, 'federated': federated}
        write_json(output / 'measurements.json', report)
        pd.DataFrame(dp['ledger']).to_csv(output / 'privacy_budget.csv', index=False, lineterminator='\n')
        graphics(output, anonymity, dp)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse', action='store_true', help='использовать локальные результаты успешных этапов')
    parser.add_argument('--output', type=Path, default=WORK / 'results')
    args = parser.parse_args()
    experiment = Experiment(reuse=args.reuse)
    pii = experiment.pii()
    anonymity = experiment.anonymity(pii)
    synthetic, samples = experiment.synthesize()
    dp, candidate = experiment.differentially_private()
    audited = experiment.evaluate(dp, candidate, samples)
    result = experiment.finish(pii, anonymity, synthetic, dp, audited, args.output)
    print(f"Гейты выпуска: {result['release']['passed']}. Результаты: {args.output}")


if __name__ == '__main__':
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    main()
