# Horizon-5 sweep history

Grids used to live as `v1`–`v5` in `horizon_sweep.py`. From here on, `TRIALS` in that file is the next run only. Each sweep writes its `comparison.txt` here under `### <sweep_id>` when it finishes.

Pick: Kendall tau, top-20 Jaccard (mean **and** min), pred std, Haaland / Thiago / Fernandes shape. Val MAE is keep/reject (~1.53 good, ~1.58+ reject), not the pick.

Current bake in `learning.py`: 1.5x width, 3 layers, Dropout 0.1/0.05, **L2 2.25e-3**, LR 5e-4, AdamW **wd 0**.

---

## Untouched (or not on the current body)

Tried hard: LR, L2, AdamW wd, dropout rates, Adam vs AdamW, width 0.5 / 1.5 / 2.0, depth 1–4 on **older** recipes.

Never or barely tried on the current body:

| Knob | Default | Notes |
| --- | --- | --- |
| `PL_BATCH_SIZE` | 32 | Never swept. Next run 16 and 64. |
| `PL_N_HIDDEN=4` | 3 | v1 14 Jaccard-trapped at 1x width (ended at 2 units). v3 12 mixed 4L with LR 1e-3. Next run retries 33-16-8-4 on this body. |
| `PL_REDUCE_LR_FACTOR` | 0.5 | Never swept. Next run 0.3. |
| `PL_REDUCE_LR_MIN` | 1e-5 | Never swept. |
| `PL_REDUCE_LR_PATIENCE` | 8 | Only as a bundle with patience 40. |
| `PL_HUBER_DELTA` | 1.8 | 1.0 lost in v1 / v3 (bundled with patience). |
| `PL_VAL_MAE_KEEP_RATIO` | 1.08 | Only vs `PL_AVERAGE_ALL_REPS`. |
| MAE loss | huber | Historic weak-L2 recipe only. |

Do not retry: dropout first-layer only, L2 3e-3 with wd 1e-4, LR 7.5e-4 stacked with wd 0, 2.0x width, Adam+L2 2.5e-3.

---

## Runs

### 20260817_134204 (was v1)

Control: option-6 net (lr 5e-4, L2 1e-3, 2 hidden, width 1x, patience 25). Winner **09_old_capacity_1.5x_3layer** (tau 0.902, J 0.751, pstd 0.092, H 31.5). Trial 14 four-layer: best tau 0.921, Jaccard trap 0.43.

```
trial                             tau      j20     pstd     H_pts
04_lr_1e-4                     0.7427   0.5316   0.2209    29.6
05_l2_weaker_1e-4              0.6571   0.4516    0.315    31.9
06_l2_stronger_5e-3            0.8125   0.6321   0.2119    29.0
07_wd_stronger_1e-3            0.8115   0.7029   0.1791    31.4
08_smaller_net_0.5x            0.7939   0.6227   0.1984    29.9
09_old_capacity_1.5x_3layer    0.9018    0.751   0.0924    31.5
10_patience_40                 0.8355   0.6675   0.1595    31.8
11_huber_1.0                   0.8289   0.6438   0.1784    29.0
12_one_hidden_layer            0.7664    0.515    0.214    30.4
13_three_hidden_layers         0.8762    0.757   0.1038    30.6
14_four_hidden_layers          0.9213   0.4261   0.3766    31.2
15_historic_pre_option6        0.5778   0.4001   0.4443    28.8
16_historic_dropout            0.8546    0.685   0.1464    29.0
17_historic_mae_loss           0.5543   0.3366   0.4726    28.2
19_adam_not_adamw              0.8417   0.6734   0.1801    31.0
20_adamw_wd_0                  0.8492   0.6774    0.129    31.0
```

### 20260818_015910 (was v2)

Control: v1 09. Winner **09_dropout_light** (tau 0.924, J 0.897, pstd 0.065, H 31.5). L2 2e-3 was close (tau 0.912, J 0.813, H 32.6).

```
trial                             tau      j20     pstd     H_pts
01_control_09                  0.9001   0.7628   0.1013    29.3
02_lr_1e-3                     0.8968    0.796   0.0914    32.3
03_1.5x_2layer                 0.8681   0.7569   0.1222    32.1
04_2.0x_3layer                 0.8928   0.7537   0.1121    32.4
05_l2_3e-4                     0.7932   0.5821    0.236    31.8
06_l2_2e-3                     0.9121   0.8125   0.0844    32.6
07_wd_1e-3                     0.8765   0.6696   0.1132    31.5
08_patience_40                 0.8719   0.7671   0.1163    31.4
09_dropout_light               0.9243    0.897   0.0646    31.5
```

### 20260818_090120 + 111528 + 124458 (was v3, three resumes)

Control: v2 09_dropout_light. Winner **02_l2_2e-3** (tau 0.964, J 0.924, min J 0.905, pstd 0.050, H 32.5, val MAE 1.543). 08 wd 0 had best Jaccard 0.962 but Haaland 31.3.

```
trial                             tau      j20     pstd     H_pts
01_control_dropout             0.9341   0.8623   0.0669    32.4
02_l2_2e-3                     0.9637   0.9238   0.0497    32.5
03_lr_1e-3                     0.9263   0.9429   0.0891    31.8
04_drop_0.2_0.1                0.9271   0.8552   0.0950    31.5
05_drop_0.05_0.05              0.9251   0.8701   0.0762    32.1
06_drop_first_only             0.9549   0.9074   0.0702    32.2
07_l2_1.5e-3                   0.9299   0.8393   0.0785    31.6
08_wd_0                        0.9248   0.9619   0.0750    31.3
09_drop_0.15_0.08              0.9364   0.8528   0.0891    31.1
10_l2_2e-3_lr_1e-3             0.9525   0.8908   0.0631    32.0
11_2.0x_l2_2e-3                0.9412   0.8321   0.0590    31.3
12_four_layer_lr_1e-3          0.9546   0.8450   0.0673    32.1
13_adam_lr_1e-3                0.9438   0.8465   0.0727    33.2
14_huber_1_patience_40         0.9338   0.8717   0.0776    32.6
15_historic_drop_l2_2e-3       0.9517   0.8701   0.0658    30.7
```

### 20260818_181610 (was v4)

Control: v3 02_l2_2e-3 (wd 1e-4). Winner **02_wd_0** (tau 0.951, J 0.914, **min J 0.905**, pstd 0.060, H 32.6). Control replay min J 0.739. 09 wd 0 + L2 2.5e-3 best tau 0.965 but J 0.853.

```
trial                             tau      j20     pstd     H_pts
01_control_l2_2e-3             0.9541   0.8638   0.0646    33.6
02_wd_0                        0.9512   0.9143   0.0596    32.6
03_lr_7.5e-4                   0.9592    0.811   0.0435    32.9
04_drop_first_only             0.9011   0.6694   0.2087    32.3
05_l2_2.5e-3                   0.9077   0.7696   0.1155    32.7
06_l2_3e-3                     0.9151   0.8014   0.1716    32.2
07_drop_0.05_0.05              0.9452   0.9247   0.1008    32.3
08_adam                        0.9528    0.871    0.053    32.9
09_wd_0_l2_2.5e-3              0.9652   0.8528    0.052    32.2
10_adam_l2_2.5e-3              0.9174   0.7758   0.1152    31.9
11_drop_first_l2_2.5e-3        0.9279   0.8363   0.0888    32.1
12_lr_7.5e-4_wd_0               0.922   0.7808   0.1326    32.7
```

### 20260819_000049 (was v5)

Control: v4 02_wd_0. Winner **02_l2_2.25e-3** (tau 0.973, min tau 0.964, J 0.933, min J 0.905, pstd 0.037, pred_std_max 1.34, H 32.4 #1, F #5, T #11, val MAE ~1.567). Runner-up 07_lr_6e-4 (best J 0.943, H 33.2) — stack on the new L2 next. Control replay was seed-fragile (pstd 0.136, pred_std_max 8.94). L2 2.5e-3 did not repeat its v4 tau win.

```
trial                             tau      j20     pstd     H_pts
01_control_wd_0                 0.914    0.837   0.1358    32.6
02_l2_2.25e-3                  0.9728   0.9333   0.0369    32.4
03_l2_1.75e-3                  0.9423    0.837    0.085    31.8
04_drop_0.05_0.05               0.906   0.8276   0.1767    33.1
05_drop_0.08_0.04              0.9259   0.8559      0.1    33.1
06_wd_3e-5                     0.9393   0.8458    0.071    32.7
07_lr_6e-4                     0.9507   0.9429   0.0594    33.2
08_drop_0.1_0.02               0.9415   0.7886   0.0857    33.4
09_l2_2.5e-3                   0.9135    0.811    0.098    32.2
10_drop_0.05_l2_2.25e-3        0.8759   0.6949   0.1969    33.0
11_drop_0.05_l2_2.5e-3          0.919   0.8638   0.1014    32.7
12_wd_3e-5_l2_2.25e-3          0.9086   0.7751   0.1843    33.1
```

### 20260819_085557

```
Horizon-5 hyperparameter sweep comparison
Control: 1.5x, 3 layers, Dropout 0.1/0.05, L2 2.25e-3, LR 5e-4, AdamW wd 0 (20260819_000049 trial 02).
Higher kendall/jaccard and lower pred_std = stabler ranks.
haaland_pts is 5-GW total in --horizon-only mode (not h1 pp90).

trial                             tau      j20     pstd     reps   epochs     H_rk    H_pts  testMAE
----------------------------------------------------------------------------------------------------
01_control                     0.9644   0.8717   0.0456      5/5     68.6        1     32.3   2.9871
    Replay last-run winner: L2 2.25e-3 on the wd-0 dropout body. New-seed control.
02_lr_6e-4                     0.9668   0.8892   0.0517      5/5     61.8        1     31.8   2.8006
    LR 6e-4 on the new L2. Last run 07 had the best Jaccard (0.943) at L2 2e-3; do they add?
03_batch_16                    0.9361   0.8249   0.1032      5/5     58.4        1     33.2   2.8644
    Batch 16 (was 32). Never swept. Noisier updates, more steps per epoch.
04_batch_64                    0.9317   0.8053   0.1332      5/5     60.6        1     32.1   2.8952
    Batch 64 (was 32). Never swept. Smoother gradients, fewer steps per epoch.
05_four_layers                 0.9205   0.8363   0.1683      5/5     44.2        1     32.6   2.7948
    4 hidden layers on this body (33-16-8-4). v1 14 Jaccard-trapped at 1x width ending at 2 units; v3 12 mixed 4L with LR 1e-3.
06_reduce_lr_0.3               0.9604   0.9333   0.0512      5/5     68.4        1     32.6   2.8252
    ReduceLROnPlateau factor 0.3 (was 0.5). Training-dynamics knob never swept.
```

### 20260819_121445

```
Horizon-5 hyperparameter sweep comparison
Control: 1.5x, 3 layers, Dropout 0.1/0.05, L2 2.25e-3, LR 5e-4, AdamW wd 0, ReduceLR 0.3 (20260819_085557 trial 06). 10 reps.
Higher kendall/jaccard and lower pred_std = stabler ranks.
haaland_pts is 5-GW total in --horizon-only mode (not h1 pp90).

trial                             tau      j20     pstd     reps   epochs     H_rk    H_pts  testMAE
----------------------------------------------------------------------------------------------------
01_control                     0.9054   0.8072   0.2087    10/10     48.9        1     32.3   2.9286
    Bake / this-run 06: L2 2.25e-3 + ReduceLR 0.3. Best Jaccard and min J this run; tau 0.960. 10-rep check.
02_rlr_0.5                     0.9399   0.8478   0.0977    10/10     53.7        1     32.6    2.818
    Previous bake (v5 02 / this-run 01): same body, ReduceLR 0.5. Tau held 0.973 then 0.964; min J did not (0.905 then 0.739).
03_lr_6e-4                      0.941    0.816   0.1039    10/10     62.3        1     32.6   2.8585
    This-run 02 / v5 07: LR 6e-4, ReduceLR 0.5. High tau both sweeps; Jaccard 0.943 then 0.889. Exact replay, not stacked on 0.3.
04_l2_2e-3                                                     /                                    
    v4 winner 02_wd_0: L2 2e-3, ReduceLR 0.5. min J 0.905 that run; v5 control replay was fragile (tau 0.914, pstd 0.136).
05_lr_6e-4_rlr_0.3                                             /                                    
06_huber_1.4                                                   /
```

### 20260821_211522

```
Horizon-5 hyperparameter sweep comparison
Current historical-prediction default: 1.5x, 3 layers, Dropout 0.1/0.05, L2 2.25e-3, LR 5e-4, AdamW wd 0, ReduceLR 0.3, Huber 1.8, batch 32. 5 horizon reps (same as prediction_builder --horizon-reps 5). horizon-only so tau/jaccard/Haaland compare to HISTORY.md.
Higher kendall/jaccard and lower pred_std = stabler ranks.
haaland_pts is 5-GW total in --horizon-only mode (not h1 pp90).

trial                             tau      j20     pstd     reps   epochs     H_rk    H_pts  testMAE
----------------------------------------------------------------------------------------------------
01_current_default_5rep        0.9495   0.8291   0.0687      5/5     85.4                     2.8823
    Live/historical default net on the current feature set (two-season priors, *_x_early, team xG). Same body as 20260819_121445 01_control, 5 reps instead of 10 so it matches prediction_builder.
```

### 20260822_203118

```
Horizon-5 hyperparameter sweep comparison
Form-window A/B on the current net: last-3 vs last-5 short form, both with a 7-appearance avg. 10 horizon reps averaged. Same 1.5x / 3-layer body as the historical default. horizon-only so tau/jaccard/Haaland compare to HISTORY.md.
Higher kendall/jaccard and lower pred_std = stabler ranks.
haaland_pts is 5-GW total in --horizon-only mode (not h1 pp90).
maePstd is the population std of per-rep test MAE.

trial                             tau      j20     pstd     reps   epochs     H_rk    H_pts  testMAE  maePstd
-------------------------------------------------------------------------------------------------------------
01_form_3_7                    0.9311   0.7753   0.0662    10/10     67.8                     3.0458   0.0044
    Current default: last-3 form + last-7 avg. 10 reps averaged.
02_form_5_7                    0.8922   0.6457   0.1392    10/10     50.9                     3.1207   0.0107
    Previous windows: last-5 form + last-7 avg. 10 reps averaged.
```

### 20260822_230257

```
Horizon-5 hyperparameter sweep comparison
Same form-window A/B (last-3 vs last-5 short form, both with a 7-appearance avg) with playing-time points stripped from train labels (1 for any appearance, 2 if 60+) and +2 restored on predicted pp90. 10 horizon reps averaged. Same 1.5x / 3-layer body. horizon-only so tau/jaccard/Haaland compare to HISTORY.md.
Higher kendall/jaccard and lower pred_std = stabler ranks.
haaland_pts is 5-GW total in --horizon-only mode (not h1 pp90).
maePstd is the population std of per-rep test MAE.

trial                             tau      j20     pstd     reps   epochs     H_rk    H_pts  testMAE  maePstd
-------------------------------------------------------------------------------------------------------------
01_form_3_7                     0.916   0.8839    0.049    10/10     65.7                     2.1189   0.0026
    Current default: last-3 form + last-7 avg. Playing-time strip/restore. 10 reps averaged.
02_form_5_7                                                    /
```
