# Resultados CIFAR-10

Treino: 45.000 imagens · validação: 5.000 (seleção de modelo) · teste: 10.000. 90 épocas por modelo.

## Ranking (ordenado pela acurácia de validação)

| # | modelo | tipo | parâmetros | val | teste | treino | tempo (s) |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | cnn_4conv_w64 | CNN | 1,686,794 | 91.30% | 91.14% | 99.03% | 302 |
| 2 | cnn_5conv | CNN | 1,704,458 | 91.04% | 89.60% | 97.96% | 346 |
| 3 | cnn_4conv | CNN | 423,562 | 90.42% | 89.54% | 95.65% | 316 |
| 4 | cnn_4conv_nodrop | CNN | 423,562 | 90.04% | 89.41% | 97.24% | 295 |
| 5 | cnn_4conv_adamw | CNN | 423,562 | 89.86% | 89.05% | 96.94% | 296 |
| 6 | cnn_3conv_k5 | CNN | 267,978 | 87.12% | 86.83% | 90.95% | 179 |
| 7 | cnn_3conv | CNN | 102,602 | 85.68% | 85.54% | 86.37% | 281 |
| 8 | cnn_4conv_noaug | CNN | 423,562 | 84.02% | 83.75% | 99.99% | 269 |
| 9 | cnn_2conv | CNN | 21,994 | 70.46% | 69.23% | 64.00% | 251 |
| 10 | mlp_5_1024-64 | MLP | 3,844,682 | 62.70% | 62.50% | 60.60% | 217 |
| 11 | mlp_2x2048 | MLP | 10,510,346 | 61.54% | 61.21% | 57.82% | 171 |
| 12 | mlp_3_1024-256 | MLP | 3,805,450 | 57.98% | 57.97% | 52.55% | 174 |
| 13 | mlp_1x512 | MLP | 1,578,506 | 57.44% | 56.77% | 51.01% | 124 |

## Top-2: cnn_4conv_w64 vs cnn_5conv

- Diferença de acurácia no teste: +1.54% (IC 95% bootstrap: +1.00% a +2.06%)
- Só cnn_4conv_w64 acerta: 451 imagens · só cnn_5conv acerta: 297 imagens
- Teste de McNemar (exato): p = 1.984e-08 → diferença significativa (α = 0,05)
- Parâmetros: 1,686,794 vs 1,704,458

| classe | cnn_4conv_w64 | cnn_5conv | diferença |
|---|---:|---:|---:|
| airplane | 92.6% | 91.1% | +1.5% |
| automobile | 95.6% | 94.9% | +0.7% |
| bird | 88.2% | 87.7% | +0.5% |
| cat | 81.1% | 79.7% | +1.4% |
| deer | 91.3% | 89.5% | +1.8% |
| dog | 86.4% | 84.2% | +2.2% |
| frog | 94.2% | 92.1% | +2.1% |
| horse | 93.4% | 91.3% | +2.1% |
| ship | 95.0% | 93.6% | +1.4% |
| truck | 93.6% | 91.9% | +1.7% |

## Melhor CNN vs melhor MLP

| | CNN | MLP |
|---|---:|---:|
| modelo | cnn_4conv_w64 | mlp_5_1024-64 |
| acurácia de teste | 91.14% | 62.50% |
| parâmetros | 1,686,794 | 3,844,682 |
| acurácia de treino | 99.03% | 60.60% |
| tempo de treino (s) | 302 | 217 |

A CNN acerta +28.64% a mais usando 2.3× menos parâmetros (McNemar p < 1e-10).

Figuras: `confusion_top2.png`, `per_class_top2.png`, `acc_vs_params.png`, `loss_curves.png`.
