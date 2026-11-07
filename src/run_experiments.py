"""Trains several CNN and MLP topologies on CIFAR-10 and writes a report.

Usage:
    uv run python src/run_experiments.py                 # train everything + report
    uv run python src/run_experiments.py --epochs 5      # quick run
    uv run python src/run_experiments.py --report-only   # rebuild plots from saved results
    uv run python src/run_experiments.py --merge results_90ep results_mlp400 --out results_final
                                                         # combine runs into one report

Outputs (in results/):
    results.csv / results.json   one row per model (val/test accuracy, params, time)
    predictions.npz              test-set predictions of every model
    confusion_top2.png           confusion matrices of the two best models
    per_class_top2.png           per-class accuracy of the two best models
    acc_vs_params.png            accuracy x number of parameters, CNN vs MLP
    loss_curves.png              training and validation loss per epoch, every model
    summary.md                   text report (ranking, McNemar test, CNN vs MLP)
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

from models.cnn import CNN
from models.mlp import MLP
from training import (CLASSES, accuracy, bootstrap_diff, count_params, load_cifar10,
                      mcnemar, predict, train)

# --- experiment grid ---------------------------------------------------------
# Each entry: name, model type, model kwargs, training kwargs.
EXPERIMENTS = [
    # CNN: depth
    ('cnn_2conv',        'cnn', dict(num_conv_layers=2, base_channels=32), {}),
    ('cnn_3conv',        'cnn', dict(num_conv_layers=3, base_channels=32), {}),
    ('cnn_4conv',        'cnn', dict(num_conv_layers=4, base_channels=32), {}),
    ('cnn_5conv',        'cnn', dict(num_conv_layers=5, base_channels=32), {}),
    # CNN: kernel size and width
    ('cnn_3conv_k5',     'cnn', dict(num_conv_layers=3, base_channels=32, kernel_size=5), {}),
    ('cnn_4conv_w64',    'cnn', dict(num_conv_layers=4, base_channels=64), {}),
    # CNN: training hyper-parameters (same topology as cnn_4conv)
    ('cnn_4conv_nodrop', 'cnn', dict(num_conv_layers=4, base_channels=32, dropout=0.0), {}),
    ('cnn_4conv_adamw',  'cnn', dict(num_conv_layers=4, base_channels=32),
     dict(optimizer='adamw', lr=2e-3, weight_decay=5e-2)),
    ('cnn_4conv_noaug',  'cnn', dict(num_conv_layers=4, base_channels=32), dict(use_augment=False)),
    # MLP
    ('mlp_1x512',        'mlp', dict(num_layers=1, hidden_size=[512]), dict(lr=0.01)),
    ('mlp_3_1024-256',   'mlp', dict(num_layers=3), dict(lr=0.01)),
    ('mlp_2x2048',       'mlp', dict(num_layers=2, hidden_size=[2048, 2048]), dict(lr=0.01)),
    ('mlp_5_1024-64',    'mlp', dict(num_layers=5, dropout=0.2), dict(lr=0.01)),
]

BUILDERS = {'cnn': CNN, 'mlp': MLP}

# --- chart styling (light surface, validated reference palette) -----------------
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
TYPE_COLOR = {'cnn': '#2a78d6', 'mlp': '#eb6834'}   # categorical slots 1 and 2
TOP_COLOR = ['#2a78d6', '#eb6834']
LOSS_COLOR = {'train': '#2a78d6', 'val': '#eb6834'}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    'seq_blue', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# --- training ----------------------------------------------------------------
def run_all(out, epochs, device, only=None):
    data = load_cifar10(device=device)
    x_val, y_val = data['val']
    x_te, y_te = data['test']
    results, preds = [], {}
    for name, kind, model_kw, train_kw in EXPERIMENTS:
        if only and name not in only:
            continue
        torch.manual_seed(0)
        model = BUILDERS[kind](**model_kw).to(device)
        n_params = count_params(model)
        print(f'\n== {name} ({kind}, {n_params:,} params) {model_kw} {train_kw}', flush=True)
        hist = train(model, data, epochs=epochs, **train_kw)
        p_te = predict(model, x_te)
        row = dict(name=name, type=kind, params=n_params,
                   val_acc=accuracy(model, x_val, y_val),
                   test_acc=(p_te == y_te).float().mean().item(),
                   train_acc=hist['train_acc'][-1],
                   train_time_s=round(hist['train_time_s'], 1),
                   epochs=epochs, model_kwargs=model_kw,
                   train_kwargs=train_kw, history=hist)
        print(f'   -> val {row["val_acc"]:.4f}  test {row["test_acc"]:.4f}  '
              f'({row["train_time_s"]:.0f}s)', flush=True)
        results.append(row)
        preds[name] = p_te.cpu().numpy()
        torch.save(model.state_dict(), out / 'checkpoints' / f'{name}.pt')

    preds['_labels'] = y_te.cpu().numpy()
    with open(out / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    np.savez(out / 'predictions.npz', **preds)


# --- report ------------------------------------------------------------------
def confusion(pred, y, n=10):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def plot_confusions(top, preds, y, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), facecolor=SURFACE)
    for ax, r in zip(axes, top):
        cm = confusion(preds[r['name']], y)
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=SEQ_BLUE, vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=8,
                        color='#ffffff' if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(10), CLASSES, rotation=45, ha='right')
        ax.set_yticks(range(10), CLASSES)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel('Classe prevista', color=INK_2)
        ax.set_ylabel('Classe real', color=INK_2)
        ax.set_title(f'{r["name"]}  ·  acurácia de teste {r["test_acc"]:.2%}',
                     color=INK, fontsize=11, loc='left')
    fig.suptitle('Matriz de confusão (teste, 10.000 imagens) — cor = fração da linha',
                 color=INK, fontsize=12, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_per_class(top, preds, y, path):
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis='x', visible=False)
    w = 0.38
    x = np.arange(10)
    for k, r in enumerate(top):
        p = preds[r['name']]
        acc = [(p[y == c] == c).mean() for c in range(10)]
        ax.bar(x + (k - 0.5) * (w + 0.02), acc, width=w, color=TOP_COLOR[k],
               label=r['name'], edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.set_ylabel('Acurácia por classe', color=INK_2)
    ax.set_title('Top-2 modelos: acurácia por classe (teste)', color=INK, fontsize=12,
                 loc='left', pad=28)
    ax.legend(frameon=False, labelcolor=INK, loc='lower left', bbox_to_anchor=(0, 1.0),
              ncol=2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_acc_vs_params(results, path):
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    style_axes(ax)
    for kind, label in (('cnn', 'CNN'), ('mlp', 'MLP')):
        rs = [r for r in results if r['type'] == kind]
        ax.scatter([r['params'] for r in rs], [r['test_acc'] for r in rs], s=70,
                   color=TYPE_COLOR[kind], edgecolor=SURFACE, linewidth=2, label=label, zorder=3)
        best = max(rs, key=lambda r: r['val_acc'])
        ax.annotate(f'{best["name"]}\n{best["test_acc"]:.1%} · {best["params"] / 1e6:.2f} M',
                    (best['params'], best['test_acc']), textcoords='offset points',
                    xytext=(10, -4), fontsize=9, color=INK, va='top')
    ax.set_xscale('log')
    ax.set_xlabel('Número de parâmetros (escala log)', color=INK_2)
    ax.set_ylabel('Acurácia de teste', color=INK_2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.set_title('CNN vs MLP: acurácia × número de parâmetros', color=INK, fontsize=12, loc='left')
    ax.legend(frameon=False, labelcolor=INK, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_loss_curves(results, path):
    """Small multiples: training vs validation loss per epoch, one panel per model."""
    n = len(results)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.7 * rows), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    ymax = max(max(r['history']['train_loss'] + r['history']['val_loss']) for r in results)
    for ax, (i, r) in zip(axes.flat, enumerate(results, 1)):
        style_axes(ax)
        h = r['history']
        ep = np.arange(1, len(h['val_loss']) + 1)
        ax.plot(ep, h['train_loss'], color=LOSS_COLOR['train'], linewidth=2)
        ax.plot(ep, h['val_loss'], color=LOSS_COLOR['val'], linewidth=2)
        best = int(np.argmin(h['val_loss']))
        ax.scatter([ep[best]], [h['val_loss'][best]], s=36, color=LOSS_COLOR['val'],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_ylim(0, ymax * 1.05)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title(f'#{i} {r["name"]}', color=INK, fontsize=10, loc='left')
        ax.text(0.97, 0.95, f'val {r["val_acc"]:.1%}', transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5, color=INK_2)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('Cross-entropy', color=INK_2, fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel('Época', color=INK_2, fontsize=9)
    handles = [plt.Line2D([], [], color=LOSS_COLOR[k], linewidth=2) for k in ('train', 'val')]
    fig.legend(handles, ['Loss de treino', 'Loss de validação (● = mínimo)'], loc='upper right',
               ncol=2, frameon=False, labelcolor=INK, fontsize=10)
    fig.suptitle('Curvas de loss por modelo (ordenados pelo ranking de validação)',
                 color=INK, fontsize=12, x=0.01, ha='left')
    fig.text(0.01, 0.945, 'A loss de treino é medida durante a época, com augmentation e dropout '
             'ativos — por isso pode ficar acima da de validação.', color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fmt_p(p):
    return '< 1e-10' if p < 1e-10 else f'= {p:.4g}'


def report(out):
    results = json.load(open(out / 'results.json'))
    preds = dict(np.load(out / 'predictions.npz'))
    y = preds.pop('_labels')
    results.sort(key=lambda r: r['val_acc'], reverse=True)   # model selection on VALIDATION

    with open(out / 'results.csv', 'w', newline='') as f:
        cols = ['rank', 'name', 'type', 'params', 'val_acc', 'test_acc', 'train_acc', 'train_time_s']
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        wr.writeheader()
        for i, r in enumerate(results, 1):
            wr.writerow({**r, 'rank': i})

    top = results[:2]
    a, b = (preds[r['name']] for r in top)
    mc = mcnemar(a, b, y)
    diff, lo, hi = bootstrap_diff(a, b, y)
    best_cnn = next(r for r in results if r['type'] == 'cnn')
    best_mlp = next(r for r in results if r['type'] == 'mlp')
    mc_cm = mcnemar(preds[best_cnn['name']], preds[best_mlp['name']], y)

    plot_confusions(top, preds, y, out / 'confusion_top2.png')
    plot_per_class(top, preds, y, out / 'per_class_top2.png')
    plot_acc_vs_params(results, out / 'acc_vs_params.png')
    plot_loss_curves(results, out / 'loss_curves.png')

    epoch_set = {r['epochs'] for r in results}
    L = ['# Resultados CIFAR-10', '',
         f'Treino: 45.000 imagens · validação: 5.000 (seleção de modelo) · teste: 10.000. '
         + (f'{epoch_set.pop()} épocas por modelo.' if len(epoch_set) == 1 else
            'O número de épocas varia por modelo (coluna "épocas"): cada arquitetura '
            'foi treinada até convergir.'), '',
         '## Ranking (ordenado pela acurácia de validação)', '',
         '| # | modelo | tipo | parâmetros | épocas | val | teste | treino | tempo (s) |',
         '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for i, r in enumerate(results, 1):
        L.append(f'| {i} | {r["name"]} | {r["type"].upper()} | {r["params"]:,} | {r["epochs"]} | '
                 f'{r["val_acc"]:.2%} | {r["test_acc"]:.2%} | {r["train_acc"]:.2%} | '
                 f'{r["train_time_s"]:.0f} |')
    sig = 'significativa' if mc['p_value'] < 0.05 else 'NÃO significativa'
    L += ['', f'## Top-2: {top[0]["name"]} vs {top[1]["name"]}', '',
          f'- Diferença de acurácia no teste: {diff:+.2%} (IC 95% bootstrap: {lo:+.2%} a {hi:+.2%})',
          f'- Só {top[0]["name"]} acerta: {mc["only_a_correct"]} imagens · '
          f'só {top[1]["name"]} acerta: {mc["only_b_correct"]} imagens',
          f'- Teste de McNemar (exato): p {fmt_p(mc["p_value"])} → diferença {sig} (α = 0,05)',
          f'- Parâmetros: {top[0]["params"]:,} vs {top[1]["params"]:,}', '',
          '| classe | ' + ' | '.join(r['name'] for r in top) + ' | diferença |', '|---|---:|---:|---:|']
    for c, cls in enumerate(CLASSES):
        accs = [(preds[r['name']][y == c] == c).mean() for r in top]
        L.append(f'| {cls} | {accs[0]:.1%} | {accs[1]:.1%} | {accs[0] - accs[1]:+.1%} |')
    ratio = best_mlp['params'] / best_cnn['params']
    L += ['', '## Melhor CNN vs melhor MLP', '',
          '| | CNN | MLP |', '|---|---:|---:|',
          f'| modelo | {best_cnn["name"]} | {best_mlp["name"]} |',
          f'| acurácia de teste | {best_cnn["test_acc"]:.2%} | {best_mlp["test_acc"]:.2%} |',
          f'| parâmetros | {best_cnn["params"]:,} | {best_mlp["params"]:,} |',
          f'| épocas | {best_cnn["epochs"]} | {best_mlp["epochs"]} |',
          f'| acurácia de treino | {best_cnn["train_acc"]:.2%} | {best_mlp["train_acc"]:.2%} |',
          f'| tempo de treino (s) | {best_cnn["train_time_s"]:.0f} | {best_mlp["train_time_s"]:.0f} |', '',
          f'A CNN acerta {best_cnn["test_acc"] - best_mlp["test_acc"]:+.2%} a mais usando '
          f'{ratio:.1f}× {"menos" if ratio > 1 else "mais"} parâmetros '
          f'(McNemar p {fmt_p(mc_cm["p_value"])}).', '',
          'Figuras: `confusion_top2.png`, `per_class_top2.png`, `acc_vs_params.png`, `loss_curves.png`.']
    (out / 'summary.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def merge(dirs, out):
    """Combines the results of several runs into `out`; later dirs win on name clashes."""
    rows, preds, labels = {}, {}, None
    for d in map(Path, dirs):
        for r in json.load(open(d / 'results.json')):
            r['source'] = str(d)
            rows[r['name']] = r
        p = dict(np.load(d / 'predictions.npz'))
        lab = p.pop('_labels')
        if labels is not None and not np.array_equal(labels, lab):
            raise ValueError(f'{d}: test labels differ from the previous runs')
        labels = lab
        preds.update(p)
    for name, r in rows.items():
        print(f'  {name:20s} {r["epochs"]:4d} epochs  <- {r["source"]}')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'results.json', 'w') as f:
        json.dump(list(rows.values()), f, indent=2)
    np.savez(out / 'predictions.npz', _labels=labels, **{k: preds[k] for k in rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results')
    ap.add_argument('--only', nargs='*', help='train only these experiment names')
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--merge', nargs='+', metavar='DIR',
                    help='combine the results of these run folders into --out and report '
                         '(later folders win when a model appears twice)')
    args = ap.parse_args()

    out = Path(args.out)
    if args.merge:
        merge(args.merge, out)
    else:
        (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    if not args.report_only and not args.merge:
        run_all(out, args.epochs, args.device, args.only)
    kinds = {r['type'] for r in json.load(open(out / 'results.json'))}
    if kinds == {'cnn', 'mlp'}:
        report(out)
    else:
        print(f'\nResults saved to {out}/. The report needs both CNNs and MLPs; '
              f'this run only has {sorted(kinds)}, so it was skipped.')


if __name__ == '__main__':
    main()
