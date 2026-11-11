"""Графики разведочного анализа и сравнения стратегий обучения."""

import matplotlib.pyplot as plt


def plot_hospital(out, eda, interpretation, hours):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    bars = axes[0].bar(['Нет', 'Да'], [eda['negative'], eda['positive']], color=['#236781', '#c3743f'])
    axes[0].bar_label(bars, padding=4, fmt='%.0f')
    axes[0].set(ylabel='Число записей', title='Повторная госпитализация')
    axes[0].set_ylim(0, eda['negative'] * 1.18)
    corr = sorted(eda['feature_target_correlation'].items(), key=lambda item: abs(item[1]), reverse=True)[:6][::-1]
    axes[1].barh([key for key, _ in corr], [value for _, value in corr], color='#236781')
    axes[1].axvline(0, color='#535b60', linewidth=.8)
    axes[1].set(xlabel='Корреляция Пирсона', title='Признаки и целевой класс')
    fig.tight_layout()
    fig.savefig(out / 'eda.png', dpi=145)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    importance = sorted(interpretation.items(), key=lambda item: item[1], reverse=True)[:7][::-1]
    axes[0].barh([name for name, _ in importance], [value for _, value in importance], color='#236781')
    axes[0].set(xlabel='Средний |SHAP|, логит', title='Вклад признаков')
    for key, label, color in (('batch_accuracy', 'Batch', '#236781'), ('online_accuracy', 'Online', '#c3743f')):
        axes[1].plot([r['hour'] for r in hours], [100 * r[key] for r in hours], label=label, color=color, linewidth=1.5)
    axes[1].set(xlabel='Час симуляции', ylabel='Accuracy, %', title='Качество на фиксированной выборке', xticks=[1, 6, 12, 18, 24])
    axes[1].grid(axis='y', alpha=.2)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / 'learning.png', dpi=145)
    plt.close(fig)
