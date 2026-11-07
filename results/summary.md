# Resultados CIFAR-10

Treino: 45.000 imagens · validação: 5.000 (seleção de modelo) · teste: 10.000. 30 épocas por modelo.

## Ranking (ordenado pela acurácia de validação)

| # | modelo | tipo | parâmetros | val | teste | treino | tempo (s) |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | cnn_4conv_w64 | CNN | 1,686,794 | 89.88% | 90.08% | 95.12% | 98 |
| 2 | cnn_5conv | CNN | 1,704,458 | 88.56% | 87.74% | 93.21% | 110 |
| 3 | cnn_4conv_nodrop | CNN | 423,562 | 88.54% | 87.95% | 92.75% | 96 |
| 4 | cnn_4conv_adamw | CNN | 423,562 | 88.32% | 87.09% | 90.92% | 105 |
| 5 | cnn_4conv | CNN | 423,562 | 87.62% | 87.31% | 90.85% | 97 |
| 6 | cnn_3conv_k5 | CNN | 267,978 | 84.16% | 84.93% | 85.46% | 84 |
| 7 | cnn_4conv_noaug | CNN | 423,562 | 83.94% | 83.28% | 99.92% | 89 |
| 8 | cnn_3conv | CNN | 102,602 | 81.96% | 81.76% | 80.48% | 84 |
| 9 | cnn_2conv | CNN | 21,994 | 65.52% | 64.75% | 58.51% | 71 |
| 10 | mlp_2x2048 | MLP | 10,510,346 | 55.18% | 55.34% | 51.10% | 59 |
| 11 | mlp_5_1024-64 | MLP | 3,844,682 | 54.48% | 55.47% | 51.10% | 72 |
| 12 | mlp_1x512 | MLP | 1,578,506 | 54.22% | 53.99% | 48.07% | 45 |
| 13 | mlp_3_1024-256 | MLP | 3,805,450 | 52.68% | 52.47% | 47.07% | 58 |

## Top-2: cnn_4conv_w64 vs cnn_5conv

- Diferença de acurácia no teste: +2.34% (IC 95% bootstrap: +1.84% a +2.93%)
- Só cnn_4conv_w64 acerta: 528 imagens · só cnn_5conv acerta: 294 imagens
- Teste de McNemar (exato): p < 1e-10 → diferença significativa (α = 0,05)
- Parâmetros: 1,686,794 vs 1,704,458

| classe | cnn_4conv_w64 | cnn_5conv | diferença |
|---|---:|---:|---:|
| airplane | 91.8% | 91.1% | +0.7% |
| automobile | 94.8% | 94.4% | +0.4% |
| bird | 86.0% | 84.4% | +1.6% |
| cat | 79.4% | 72.0% | +7.4% |
| deer | 92.4% | 90.0% | +2.4% |
| dog | 83.5% | 79.9% | +3.6% |
| frog | 93.8% | 90.2% | +3.6% |
| horse | 91.8% | 90.5% | +1.3% |
| ship | 93.4% | 93.5% | -0.1% |
| truck | 93.9% | 91.4% | +2.5% |

## Melhor CNN vs melhor MLP

| | CNN | MLP |
|---|---:|---:|
| modelo | cnn_4conv_w64 | mlp_2x2048 |
| acurácia de teste | 90.08% | 55.34% |
| parâmetros | 1,686,794 | 10,510,346 |
| acurácia de treino | 95.12% | 51.10% |
| tempo de treino (s) | 98 | 59 |

A CNN acerta +34.74% a mais usando 6.2× menos parâmetros (McNemar p < 1e-10).

Figuras: `confusion_top2.png`, `per_class_top2.png`, `acc_vs_params.png`, `loss_curves.png`.
