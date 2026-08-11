"""
recent500_vs_collapse 그림 재생성 스크립트
- 기존 그림의 문제 수정: (1) StreamingLLM 범례 마커 누락 (2) StreamingLLM이 OURS에 가려짐
- 데이터 소스: results/M1_final_charrep_20260808_010351.csv (80% budget만 필터링)
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('results/legacy_pre_fix/M1_final_charrep_20260808_010351.csv')
df80 = df[df['budget'] == 0.8].copy()

METHOD_STYLE = {
    'streaming': {'marker': 'D', 'color': '#1f77b4', 'label': 'StreamingLLM', 'size': 90, 'zorder': 4, 'alpha': 0.85, 'edgecolor': 'black', 'lw': 0.6},
    'h2o':       {'marker': 'o', 'color': '#4c72b0', 'label': 'H2O',          'size': 70, 'zorder': 2, 'alpha': 0.75, 'edgecolor': 'none',  'lw': 0},
    'snapkv':    {'marker': 's', 'color': '#dd8452', 'label': 'SnapKV',       'size': 70, 'zorder': 2, 'alpha': 0.75, 'edgecolor': 'none',  'lw': 0},
    'pyramidkv': {'marker': '^', 'color': '#55a868', 'label': 'PyramidKV',    'size': 70, 'zorder': 2, 'alpha': 0.75, 'edgecolor': 'none',  'lw': 0},
    'adakv':     {'marker': 'd', 'color': '#c44e52', 'label': 'AdaKV',        'size': 70, 'zorder': 2, 'alpha': 0.75, 'edgecolor': 'none',  'lw': 0},
    'ours':      {'marker': '*', 'color': '#8172b2', 'label': 'OURS (HSS)',   'size': 130,'zorder': 3, 'alpha': 0.85, 'edgecolor': 'none',  'lw': 0},
}
DRAW_ORDER = ['h2o', 'snapkv', 'pyramidkv', 'adakv', 'ours', 'streaming']

tasks = ['qmsum', 'gov_report']
task_titles = {'qmsum': 'QMSum', 'gov_report': 'GovReport'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, task in zip(axes, tasks):
    sub = df80[df80['task'] == task]
    # 상관계수는 5개 top-k 스코어링 방법 기준(StreamingLLM은 sink 내장 특수 케이스라 제외,
    # 논문 본문 수치 QMSum r=-0.676/GovReport r=-0.184와 정합성 유지)
    corr_sub = sub[sub['method'] != 'streaming']
    r, p = stats.pearsonr(corr_sub['recent500_coverage'], corr_sub['word_rep'])

    for method in DRAW_ORDER:
        m_sub = sub[sub['method'] == method]
        style = METHOD_STYLE[method]
        ax.scatter(
            m_sub['recent500_coverage'], m_sub['word_rep'],
            marker=style['marker'], c=style['color'], s=style['size'],
            alpha=style['alpha'], zorder=style['zorder'],
            edgecolors=style['edgecolor'], linewidths=style['lw'],
            label=style['label'],
        )

    ax.axhline(y=0.3, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(sub['recent500_coverage'].min() + 2, 0.32, 'collapse threshold', fontsize=9, color='gray')
    ax.set_title(f"{task_titles[task]}  (r = {r:.3f}, p {'< 0.001' if p < 0.001 else f'= {p:.3f}'})", fontsize=13, fontweight='bold')
    ax.set_xlabel('Recent-500 Token Coverage\n(# of last 500 original positions retained)')
    ax.set_ylabel('Repetition Ratio (trigram)')
    ax.set_xlim(360, 505)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
legend_order = ['StreamingLLM', 'H2O', 'SnapKV', 'PyramidKV', 'AdaKV', 'OURS (HSS)']
ordered = [(handles[labels.index(l)], l) for l in legend_order]
fig.legend([h for h, l in ordered], [l for h, l in ordered],
           loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.02), fontsize=11, frameon=False)

fig.suptitle('Recent-Context Retention vs. Repetition Collapse\n(Qwen3-4B, 80% Budget, 50 samples/method, seed=42)',
             fontsize=15, y=1.03)

plt.tight_layout()
import os
os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/legacy_pre_fix/recent500_vs_collapse_v3.png', dpi=150, bbox_inches='tight')
print('저장 완료: results/figures/legacy_pre_fix/recent500_vs_collapse_v3.png')
