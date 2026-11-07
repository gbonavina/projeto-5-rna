"""Plots one example image of each CIFAR-10 class.

Usage:
    uv run python src/plot_samples.py                 # first image of each class
    uv run python src/plot_samples.py --seed 3        # random image of each class
    uv run python src/plot_samples.py --interp nearest  # show the raw pixels
    uv run python src/plot_samples.py --compact         # small, crisp 3x thumbnails
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from training import CLASSES

SURFACE, INK, INK_2 = '#fcfcfb', '#0b0b0b', '#52514e'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data')
    ap.add_argument('--seed', type=int, default=None,
                    help='pick a random image per class (default: the first one)')
    ap.add_argument('--interp', default='bicubic',
                    help="matplotlib interpolation: 'bicubic' (smooth) or 'nearest' (raw pixels)")
    ap.add_argument('--compact', action='store_true',
                    help='one row of small thumbnails, each at exactly 3x (96x96 px)')
    ap.add_argument('--out', default='results/samples_per_class.png')
    args = ap.parse_args()

    ds = torchvision.datasets.CIFAR10(root=args.root, train=True, download=False)
    labels = np.array(ds.targets)
    rng = np.random.default_rng(args.seed) if args.seed is not None else None

    if args.compact:
        # each image occupies exactly 96 px (32 px * 3) at dpi=100 -> crisp integer upscale
        dpi, size, gap, top = 100, 0.96, 0.14, 0.75
        fig_w = 10 * size + 11 * gap
        fig, axes = plt.subplots(1, 10, figsize=(fig_w, size + top + 0.08), dpi=dpi, facecolor=SURFACE)
        fig.subplots_adjust(left=gap / fig_w, right=1 - gap / fig_w, bottom=0.08 / (size + top + 0.08),
                            top=size / (size + top + 0.08) + 0.08 / (size + top + 0.08),
                            wspace=gap / size)
    else:
        dpi = 150
        fig, axes = plt.subplots(2, 5, figsize=(11, 5.6), facecolor=SURFACE)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.01, wspace=0.08, hspace=0.25)
    for c, ax in enumerate(axes.flat):
        idx = np.flatnonzero(labels == c)
        i = rng.choice(idx) if rng is not None else idx[0]
        ax.imshow(ds.data[i], interpolation='nearest' if args.compact else args.interp)   # raw 32x32 RGB, no normalization
        ax.set_title(CLASSES[c] if args.compact else f'{c} · {CLASSES[c]}', color=INK,
                     fontsize=9 if args.compact else 11)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle('CIFAR-10: um exemplo de cada classe', color=INK,
                 fontsize=11 if args.compact else 13, x=0.01, y=0.97 if args.compact else 0.98, ha='left')
    if args.compact:
        how = 'ampliadas exatamente 3× (96×96 px), sem suavização'
    elif args.interp == 'nearest':
        how = 'ampliadas sem suavização'
    else:
        how = f'ampliadas com interpolação {args.interp}'
    fig.text(0.01, 0.82 if args.compact else 0.9,
             f'Imagens originais de 32×32 pixels (conjunto de treino), {how}.',
             color=INK_2, fontsize=8 if args.compact else 9)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=dpi, facecolor=SURFACE)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
