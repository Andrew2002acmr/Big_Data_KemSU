"""Приёмка заранее выбранного выпуска OpenDP с бюджетом ε=3."""

from evaluation import FAIRNESS_LIMIT, attack_audit, dcr_audit, fairness, fidelity, utility
from mock_insurance import AGES, DISTRICTS, PRODUCTS, STATUSES, TARGET
from synthesis import verify_synthesis

RELEASE_EPSILON = 3


def gates(train, test, control, release, raw_tables, baseline, ledger):
    quality = fidelity(train, release)
    tstr, prediction = utility(release, test)
    fair = fairness(test, prediction)
    integrity = verify_synthesis(release, raw_tables, 'release')
    quasi = ['age_band', 'district', 'product', 'policy_year']
    release_counts = release.groupby(quasi, observed=True).size()
    reference_counts = train.groupby(quasi, observed=True).size().reindex(release_counts.index).fillna(0)
    rare_mapping = int(((reference_counts > 0) & (reference_counts <= 2)).sum())
    groups = {
        'ages': set(AGES) <= set(release.age_band), 'districts': set(DISTRICTS) <= set(release.district),
        'products': set(PRODUCTS) <= set(release['product']), 'statuses': set(STATUSES) <= set(release.status),
        'claims_in_every_status': bool(release.groupby('status')[TARGET].nunique().eq(2).all()),
    }
    distances = dcr_audit(train, control, release)
    attacks = attack_audit(train, control, release)
    relative = tstr['auc'] / baseline
    assessment = {
        'pii_overlap': {'value': integrity['pii_overlap'], 'passed': integrity['pii_overlap'] == 0},
        'rare_mapping': {'value': rare_mapping, 'passed': rare_mapping == 0},
        **{name: {'value': attack['main'], 'limit': attack['control_ci95'][1],
                  'passed': attack['main'] <= attack['control_ci95'][1]} for name, attack in attacks.items()},
        'utility': {'value': relative, 'limit': .90, 'passed': relative >= .90},
        'coverage': {'value': groups, 'passed': all(groups.values())},
        'fairness': {'value': fair['tpr_gap'], 'limit': FAIRNESS_LIMIT,
                     'passed': fair['tpr_gap'] <= FAIRNESS_LIMIT and min(g['positives'] for g in fair['groups'].values()) >= 25},
        'dp_report': {'value': {'epsilon': sum(row['epsilon_upper'] for row in ledger),
                                'delta': sum(row['delta'] for row in ledger), 'unit': ledger[0]['unit'],
                                'accountant': ledger[0]['accountant']},
                      'passed': bool(ledger) and sum(row['epsilon_upper'] for row in ledger) <= RELEASE_EPSILON + 1e-9},
    }
    return {'method': 'OpenDP', 'epsilon': RELEASE_EPSILON, 'gates': assessment,
            'passed': all(row['passed'] for row in assessment.values()), 'tstr': tstr, 'fairness': fair,
            'fidelity': quality, 'attacks': attacks, 'distances': distances, 'relations': integrity,
            'minimum_release_equivalence_class': int(release_counts.min()),
            'release_groups_le_2': int(release_counts.le(2).sum())}
