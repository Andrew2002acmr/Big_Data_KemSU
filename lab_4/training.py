"""Пакетная модель, онлайн-обновления и жизненный цикл MLflow."""

import json
from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from data import FEATURES, SEED

MODEL = 'HospitalReadmission'
EXPERIMENT = 'Риск повторной госпитализации'


def configure_tracking(work):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    uri = 'sqlite:///' + (work / 'mlflow.db').as_posix()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(work / 'artifacts').as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow.MlflowClient()


def preprocess():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler())


def metrics(y, probability, threshold):
    prediction = probability >= threshold
    negative = np.asarray(y) == 0
    return {
        'auc': float(roc_auc_score(y, probability)), 'ap': float(average_precision_score(y, probability)),
        'accuracy': float(accuracy_score(y, prediction)), 'f1': float(f1_score(y, prediction, zero_division=0)),
        'recall': float(recall_score(y, prediction, zero_division=0)),
        'fpr': float(prediction[negative].mean()),
    }


def operating_threshold(y, probability):
    negative = np.sort(probability[np.asarray(y) == 0])
    allowed = int(len(negative) * .007)
    return float(np.nextafter(negative[-max(allowed, 1)], np.inf))


def assess(model, validation):
    x, y = validation
    proba = model.predict_proba(x)[:, 1]
    threshold = operating_threshold(y, proba)
    return threshold, metrics(y, proba, threshold)


def fit_batch(x, y):
    return Pipeline([('prep', preprocess()), ('classifier', LogisticRegression(max_iter=500))]).fit(x, y)


def compare_models(train, validation):
    x, y = train
    models = {
        'Логистическая регрессия': LogisticRegression(max_iter=500),
        'Случайный лес': RandomForestClassifier(n_estimators=120, min_samples_leaf=24, random_state=SEED, n_jobs=2),
        'Градиентный бустинг': HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, random_state=SEED),
    }
    comparisons = {}
    for name, classifier in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({'model': type(classifier).__name__, 'train_rows': len(x), 'seed': SEED})
            start = perf_counter()
            with threadpool_limits(limits=2):
                model = Pipeline([('prep', preprocess()), ('classifier', classifier)]).fit(x, y)
                elapsed = perf_counter() - start
                threshold, result = assess(model, validation)
            mlflow.log_params({'threshold': threshold})
            mlflow.log_metrics({**result, 'fit_sec': elapsed})
            comparisons[name] = {**result, 'fit_sec': elapsed, 'threshold': threshold}
            if isinstance(classifier, LogisticRegression):
                batch = model
    return batch, comparisons


def replay_online(train, validation, batch):
    x, y = train
    x_val, y_val = validation
    warm_count = len(x) // 3
    online_prep = preprocess().fit(x.iloc[:warm_count])
    learner = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=.005,
                            alpha=.001, average=True, random_state=SEED)
    warm_batch = fit_batch(x.iloc[:warm_count], y.iloc[:warm_count])
    started = perf_counter()
    learner.fit(online_prep.transform(x.iloc[:warm_count]), y.iloc[:warm_count])
    warm_seconds = perf_counter() - started
    online = Pipeline([('prep', online_prep), ('classifier', learner)])
    stable_prob = warm_batch.predict_proba(x_val)[:, 1]
    warm_threshold = operating_threshold(y_val, stable_prob)
    stable_accuracy = metrics(y_val, stable_prob, warm_threshold)['accuracy']
    rows = np.arange(warm_count, len(x))
    chunks = np.array_split(rows, 24)
    arrival_hour = np.repeat(np.arange(1, 25), [len(chunk) for chunk in chunks])
    label_delay = np.random.default_rng(SEED).integers(0, 4, size=len(rows))
    discharge_hour = arrival_hour - 30 * 24 - label_delay
    hours = []
    with mlflow.start_run(run_name='24 часа: online partial_fit'):
        mlflow.log_params({'warm_rows': warm_count, 'label_delay_days': 30, 'updates': 24, 'eta0': .005})
        for hour in range(1, 25):
            ready = (arrival_hour == hour) & (discharge_hour + 30 * 24 <= hour)
            positions = rows[ready]
            started = perf_counter()
            learner.partial_fit(online_prep.transform(x.iloc[positions]), y.iloc[positions])
            update_sec = perf_counter() - started
            threshold, score = assess(online, validation)
            batch_score = assess(batch, validation)[1]['accuracy'] if hour == 24 else stable_accuracy
            hours.append({'hour': hour, 'rows': len(positions), 'late_labels': int((label_delay[ready] > 0).sum()),
                          'update_sec': update_sec,
                          'online_accuracy': score['accuracy'], 'batch_accuracy': batch_score,
                          'online_auc': score['auc']})
            mlflow.log_metrics({'online_accuracy': score['accuracy'], 'online_auc': score['auc'],
                                'batch_accuracy': batch_score, 'update_sec': update_sec}, step=hour)
        mlflow.log_metric('warm_fit_sec', warm_seconds)
    return online, hours, warm_seconds


def baseline(model, background):
    return model.named_steps['prep'].transform(background).mean(axis=0).tolist()


def shap_linear(model, frame, background_mean):
    """Точные interventional SHAP в шкале логита для линейной модели."""
    features = model.named_steps['prep'].transform(frame)
    classifier = model.named_steps['classifier']
    weights = classifier.coef_[0]
    base = float(classifier.intercept_[0] + np.dot(background_mean, weights))
    contributions = (features - np.asarray(background_mean)) * weights
    return base, contributions


def register(client, model, x, threshold, run_name, evaluation, fit_seconds=None):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({'threshold': threshold, 'train_rows': len(x), 'features': len(FEATURES)})
        mlflow.log_metrics(evaluation)
        if fit_seconds is not None:
            mlflow.log_metric('fit_sec', fit_seconds)
        signature = infer_signature(x.head(5), model.predict(x.head(5)))
        info = mlflow.sklearn.log_model(model, name='model', signature=signature,
                                      input_example=x.head(3).reset_index(drop=True))
        version = mlflow.register_model(info.model_uri, MODEL)
    meta = {'threshold': threshold, 'baseline': baseline(model, x), 'validation_auc': evaluation['auc']}
    client.set_model_version_tag(MODEL, version.version, 'serving', json.dumps(meta))
    return str(version.version)


def get_version(client, alias):
    version = client.get_model_version_by_alias(MODEL, alias)
    model = mlflow.sklearn.load_model(f'models:/{MODEL}/{version.version}')
    return model, str(version.version), json.loads(version.tags['serving'])
