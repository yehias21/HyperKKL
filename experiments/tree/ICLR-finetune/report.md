# ICLR-finetune — Row 6: Phase 2 with the base maps fine-tuned jointly (unfreeze_base) vs frozen (Dynamic)

*parent* `ICLR` · *contribution* iclr · *status* done · jobs 12/12

**Question.** Does joint fine-tuning of the base maps in Phase 2 help? (reviewer NpQ8)

**Decision rule.** Paired relative NRMSE vs dynamic (frozen) on id and on fam_zero (fine-tuning breaks exact zero-input consistency).

# Metric `nrmse`

Definitions in hyperkkl/metrics.py. Mean over training seeds with a Student-t 95% CI over seeds (single-seed nodes: episode bootstrap, which ignores training variability); lower is better. Paired cells: mean per-seed relative change vs the reference, negative = better, bold = CI excludes 0. µs/step: CPU, batch 1.

## Scoreboard (id / clean, all families)

| method | iclr_duffing | iclr_vdp | iclr_rossler | iclr_lorenz |
|---|---|---|---|---|
| autonomous | 0.2079 | 0.0637 | 0.1833 | 0.0171 |
| dynamic | 0.1340 | 0.0198 | 0.1038 | 0.0169 |
| dynamic[unfreeze_base=true] | 0.1231 | 0.0149 | 0.0888 | 0.0153 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.1335 | 0.0173 | 0.0909 | 0.0161 |

Paired vs `dynamic` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | iclr_duffing | iclr_vdp | iclr_rossler | iclr_lorenz |
|---|---|---|---|---|
| autonomous | 0/10 (+55%*) | 0/10 (+223%*) | 0/9 (+77%*) | 0/8 (+1%*) |
| dynamic[unfreeze_base=true] | 6/0 (-8%*) | 10/0 (-24%*) | 10/0 (-14%*) | 1/0 (-10%) |
| dynamic[unfreeze_base=true,lr=0.0005] | 0/0 (-0%) | 8/0 (-13%*) | 10/0 (-12%*) | 10/0 (-5%*) |

## Scoreboard (shape_ood / clean, all families)

| method | iclr_duffing | iclr_vdp | iclr_rossler | iclr_lorenz |
|---|---|---|---|---|
| autonomous | 0.1828 | 0.0604 | 0.1618 | 0.0173 |
| dynamic | 0.1174 | 0.0201 | 0.1091 | 0.0171 |
| dynamic[unfreeze_base=true] | 0.1072 | 0.0146 | 0.0930 | 0.0155 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.1170 | 0.0173 | 0.0953 | 0.0163 |

Paired vs `dynamic` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | iclr_duffing | iclr_vdp | iclr_rossler | iclr_lorenz |
|---|---|---|---|---|
| autonomous | 0/4 (+56%*) | 0/4 (+201%*) | 0/4 (+48%*) | 0/4 (+1%*) |
| dynamic[unfreeze_base=true] | 3/0 (-9%*) | 4/0 (-27%*) | 4/0 (-15%*) | 0/0 (-9%) |
| dynamic[unfreeze_base=true,lr=0.0005] | 0/0 (-0%) | 3/0 (-14%*) | 4/0 (-13%*) | 4/0 (-5%*) |

## Scoreboard (sequence_ood / clean, all families)

| method | iclr_duffing | iclr_vdp | iclr_rossler | iclr_lorenz |
|---|---|---|---|---|
| autonomous | 0.1383 | 0.0840 | 0.2265 | 0.0176 |
| dynamic | 0.0878 | 0.0256 | 0.1144 | 0.0172 |
| dynamic[unfreeze_base=true] | 0.0912 | 0.0164 | 0.0891 | 0.0157 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0912 | 0.0202 | 0.0913 | 0.0164 |

Paired vs `dynamic` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | iclr_duffing | iclr_vdp | iclr_rossler | iclr_lorenz |
|---|---|---|---|---|
| autonomous | 0/1 (+58%*) | 0/1 (+230%*) | 0/1 (+98%*) | 0/1 (+2%*) |
| dynamic[unfreeze_base=true] | 0/0 (+4%) | 1/0 (-35%*) | 1/0 (-22%*) | 0/0 (-9%) |
| dynamic[unfreeze_base=true,lr=0.0005] | 0/0 (+4%) | 1/0 (-21%*) | 1/0 (-20%*) | 1/0 (-5%*) |

## iclr_duffing

phase 1: eps_enc_rel 0.00696, eps_dec 0.0642, eps_pde_rms 0.363, eps_pde_ss_rms 0.0374, l_dec_q95 28.4, l_dec_mean 8.03 · mean job wall 37.1 min

- state coverage id: 0.0% of eval states outside the training box, 1.2% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.5% beyond the training kNN q99
- state coverage sequence_ood: 0.1% of eval states outside the training box, 14.3% beyond the training kNN q99
- state coverage fam_zero: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage fam_constant: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| autonomous | 0.2104 [0.2004, 0.2205] | 0.2389 [0.2321, 0.2457] | 0.2143 [0.2081, 0.2204] | 0.2161 [0.2093, 0.2228] | 0.1623 [0.1562, 0.1683] | 0.2508 [0.2442, 0.2575] | 0.2177 [0.2118, 0.2237] | 0.1638 [0.1562, 0.1713] | 0.1936 [0.1868, 0.2005] | 0.2147 [0.2072, 0.2222] | 0.2079 [0.2009, 0.2149] | 0.00 | – | 87.01 |
| dynamic | 0.1420 [0.1371, 0.1470] | 0.1495 [0.1471, 0.1519] | 0.1431 [0.1391, 0.1472] | 0.1426 [0.1416, 0.1437] | **0.1049 [0.0890, 0.1207]** | 0.1716 [0.1648, 0.1785] | 0.1436 [0.1417, 0.1456] | 0.0870 [0.0818, 0.0922] | 0.1252 [0.1231, 0.1273] | 0.1339 [0.1329, 0.1348] | 0.1340 [0.1311, 0.1368] | 0.00 | 3555552 | 1805.90 |
| dynamic[unfreeze_base=true] | **0.1269 [0.1147, 0.1390]** | **0.1378 [0.1248, 0.1508]** | **0.1289 [0.1175, 0.1404]** | **0.1294 [0.1169, 0.1419]** | 0.1058 [0.0743, 0.1372] | **0.1551 [0.1409, 0.1693]** | **0.1312 [0.1180, 0.1444]** | **0.0830 [0.0774, 0.0886]** | **0.1125 [0.1029, 0.1220]** | **0.1228 [0.1108, 0.1348]** | **0.1231 [0.1157, 0.1305]** | 0.00 | 3555552 | 1805.69 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.1392 [0.1288, 0.1497] | 0.1513 [0.1397, 0.1629] | 0.1399 [0.1305, 0.1493] | 0.1415 [0.1309, 0.1520] | 0.1052 [0.0955, 0.1149] | 0.1695 [0.1572, 0.1817] | 0.1444 [0.1331, 0.1557] | 0.0910 [0.0787, 0.1033] | 0.1219 [0.1130, 0.1309] | 0.1349 [0.1251, 0.1446] | 0.1335 [0.1237, 0.1433] | 0.00 | 3555552 | 1797.25 |

Paired vs `dynamic`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| autonomous | **+48.2% [+37, +60]** | **+59.8% [+58, +62]** | **+49.7% [+44, +56]** | **+51.5% [+48, +55]** | **+55.2% [+29, +82]** | **+46.2% [+38, +55]** | **+51.6% [+49, +54]** | **+88.2% [+82, +94]** | **+54.7% [+48, +61]** | **+60.4% [+56, +65]** | **+55.2% [+49, +62]** |
| dynamic[unfreeze_base=true] | -10.6% [-22, +1] | **-7.8% [-15, -0]** | **-9.9% [-20, -0]** | **-9.3% [-18, -1]** | +0.6% [-15, +16] | -9.6% [-21, +2] | **-8.7% [-17, -1]** | **-4.6% [-8, -1]** | **-10.2% [-19, -1]** | -8.3% [-17, +0] | **-8.1% [-15, -1]** |
| dynamic[unfreeze_base=true,lr=0.0005] | -1.9% [-13, +9] | +1.2% [-5, +7] | -2.3% [-10, +6] | -0.8% [-8, +6] | +0.5% [-11, +12] | -1.2% [-11, +9] | +0.5% [-6, +7] | +4.5% [-4, +13] | -2.6% [-11, +5] | +0.7% [-6, +7] | -0.3% [-8, +8] |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| autonomous | 0.1583 [0.1501, 0.1665] | 0.1994 [0.1904, 0.2084] | 0.1848 [0.1777, 0.1918] | 0.1886 [0.1794, 0.1978] | 0.1828 [0.1744, 0.1911] | 0.00 | – | 87.01 |
| dynamic | 0.0950 [0.0925, 0.0975] | 0.1302 [0.1266, 0.1339] | 0.1264 [0.1242, 0.1286] | 0.1181 [0.1172, 0.1190] | 0.1174 [0.1165, 0.1184] | 0.00 | 3555552 | 1805.90 |
| dynamic[unfreeze_base=true] | **0.0894 [0.0871, 0.0917]** | **0.1175 [0.1064, 0.1286]** | **0.1146 [0.1053, 0.1239]** | **0.1073 [0.0967, 0.1179]** | **0.1072 [0.0992, 0.1152]** | 0.00 | 3555552 | 1805.69 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0963 [0.0841, 0.1086] | 0.1290 [0.1194, 0.1386] | 0.1241 [0.1151, 0.1331] | 0.1185 [0.1082, 0.1289] | 0.1170 [0.1067, 0.1273] | 0.00 | 3555552 | 1797.25 |

Paired vs `dynamic`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| autonomous | **+66.6% [+62, +71]** | **+53.1% [+42, +64]** | **+46.2% [+39, +54]** | **+59.6% [+53, +66]** | **+55.6% [+48, +63]** |
| dynamic[unfreeze_base=true] | **-5.9% [-10, -2]** | -9.7% [-21, +1] | **-9.3% [-18, -0]** | **-9.2% [-18, -1]** | **-8.7% [-16, -1]** |
| dynamic[unfreeze_base=true,lr=0.0005] | +1.3% [-9, +12] | -0.9% [-11, +9] | -1.8% [-10, +7] | +0.3% [-8, +8] | -0.4% [-10, +9] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.1383 [0.1331, 0.1435] | 0.1383 [0.1331, 0.1435] | 0.00 | – | 87.01 |
| dynamic | **0.0878 [0.0756, 0.0999]** | **0.0878 [0.0756, 0.0999]** | 0.00 | 3555552 | 1805.90 |
| dynamic[unfreeze_base=true] | 0.0912 [0.0581, 0.1243] | 0.0912 [0.0581, 0.1243] | 0.00 | 3555552 | 1805.69 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0912 [0.0798, 0.1026] | 0.0912 [0.0798, 0.1026] | 0.00 | 3555552 | 1797.25 |

Paired vs `dynamic`:

| method | drive | all |
|---|---|---|
| autonomous | **+57.9% [+34, +82]** | **+57.9% [+34, +82]** |
| dynamic[unfreeze_base=true] | +3.6% [-21, +28] | +3.6% [-21, +28] |
| dynamic[unfreeze_base=true,lr=0.0005] | +4.0% [-11, +19] | +4.0% [-11, +19] |

### fam_zero / clean

| method | zero | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.2228 [0.2134, 0.2322] | 0.2228 [0.2134, 0.2322] | 0.00 | – | 87.01 |
| dynamic | 0.2228 [0.2134, 0.2322] | 0.2228 [0.2134, 0.2322] | 0.00 | 3555552 | 1805.90 |
| dynamic[unfreeze_base=true] | **0.1433 [0.1307, 0.1559]** | **0.1433 [0.1307, 0.1559]** | 0.00 | 3555552 | 1805.69 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.1558 [0.1453, 0.1663] | 0.1558 [0.1453, 0.1663] | 0.00 | 3555552 | 1797.25 |

Paired vs `dynamic`:

| method | zero | all |
|---|---|---|
| autonomous | +0.0% [+0, +0] | +0.0% [+0, +0] |
| dynamic[unfreeze_base=true] | **-35.7% [-40, -32]** | **-35.7% [-40, -32]** |
| dynamic[unfreeze_base=true,lr=0.0005] | **-30.1% [-32, -28]** | **-30.1% [-32, -28]** |

### fam_constant / clean

| method | constant | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.2247 [0.2174, 0.2320] | 0.2247 [0.2174, 0.2320] | 0.00 | – | 87.01 |
| dynamic | 0.1489 [0.1468, 0.1511] | 0.1489 [0.1468, 0.1511] | 0.00 | 3555552 | 1805.90 |
| dynamic[unfreeze_base=true] | **0.1358 [0.1235, 0.1480]** | **0.1358 [0.1235, 0.1480]** | 0.00 | 3555552 | 1805.69 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.1479 [0.1377, 0.1582] | 0.1479 [0.1377, 0.1582] | 0.00 | 3555552 | 1797.25 |

Paired vs `dynamic`:

| method | constant | all |
|---|---|---|
| autonomous | **+50.8% [+46, +56]** | **+50.8% [+46, +56]** |
| dynamic[unfreeze_base=true] | **-8.8% [-18, -0]** | **-8.8% [-18, -0]** |
| dynamic[unfreeze_base=true,lr=0.0005] | -0.7% [-8, +6] | -0.7% [-8, +6] |

## iclr_vdp

phase 1: eps_enc_rel 0.0189, eps_dec 2.16, eps_pde_rms 1.12, eps_pde_ss_rms 0.232, l_dec_q95 14.6, l_dec_mean 4.52 · mean job wall 30.6 min

- state coverage id: 0.0% of eval states outside the training box, 0.3% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.3% beyond the training kNN q99
- state coverage fam_zero: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage fam_constant: 0.0% of eval states outside the training box, 1.2% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| autonomous | 0.0256 [0.0255, 0.0257] | 0.0649 [0.0592, 0.0706] | 0.0782 [0.0716, 0.0847] | 0.0365 [0.0348, 0.0382] | 0.0741 [0.0692, 0.0790] | 0.0760 [0.0689, 0.0831] | 0.0669 [0.0616, 0.0721] | 0.0739 [0.0677, 0.0800] | 0.0732 [0.0673, 0.0790] | 0.0643 [0.0587, 0.0699] | 0.0637 [0.0588, 0.0686] | 0.00 | – | 90.63 |
| dynamic | 0.0202 [0.0182, 0.0222] | 0.0187 [0.0154, 0.0220] | 0.0191 [0.0162, 0.0220] | 0.0188 [0.0171, 0.0204] | 0.0227 [0.0176, 0.0278] | 0.0201 [0.0144, 0.0259] | 0.0191 [0.0171, 0.0211] | 0.0204 [0.0175, 0.0234] | 0.0207 [0.0174, 0.0241] | 0.0177 [0.0153, 0.0200] | 0.0198 [0.0168, 0.0227] | 0.00 | 3555552 | 1786.85 |
| dynamic[unfreeze_base=true] | **0.0158 [0.0155, 0.0161]** | **0.0147 [0.0145, 0.0148]** | **0.0143 [0.0137, 0.0150]** | **0.0146 [0.0141, 0.0151]** | **0.0158 [0.0151, 0.0166]** | **0.0153 [0.0148, 0.0158]** | **0.0148 [0.0148, 0.0149]** | **0.0142 [0.0134, 0.0150]** | **0.0153 [0.0150, 0.0156]** | **0.0142 [0.0140, 0.0145]** | **0.0149 [0.0146, 0.0152]** | 0.00 | 3555552 | 1805.91 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0184 [0.0174, 0.0194] | 0.0164 [0.0156, 0.0173] | 0.0159 [0.0153, 0.0165] | 0.0167 [0.0155, 0.0179] | 0.0190 [0.0161, 0.0218] | 0.0172 [0.0161, 0.0182] | 0.0172 [0.0159, 0.0185] | 0.0176 [0.0158, 0.0194] | 0.0182 [0.0165, 0.0198] | 0.0162 [0.0153, 0.0171] | 0.0173 [0.0160, 0.0186] | 0.00 | 3555552 | 1818.94 |

Paired vs `dynamic`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| autonomous | **+26.9% [+14, +40]** | **+248.8% [+160, +338]** | **+310.6% [+214, +407]** | **+94.7% [+69, +121]** | **+228.2% [+134, +323]** | **+281.9% [+142, +421]** | **+251.1% [+191, +311]** | **+262.7% [+182, +343]** | **+254.2% [+170, +338]** | **+264.8% [+186, +344]** | **+223.4% [+152, +295]** |
| dynamic[unfreeze_base=true] | **-21.6% [-31, -12]** | **-21.4% [-35, -8]** | **-24.8% [-40, -10]** | **-22.1% [-30, -15]** | **-29.8% [-50, -10]** | **-23.3% [-46, -1]** | **-22.2% [-30, -14]** | **-30.5% [-44, -17]** | **-26.0% [-39, -13]** | **-19.3% [-30, -9]** | **-24.4% [-37, -12]** |
| dynamic[unfreeze_base=true,lr=0.0005] | **-8.8% [-14, -3]** | **-11.9% [-23, -1]** | **-16.6% [-27, -6]** | **-11.2% [-14, -9]** | -16.2% [-35, +2] | -14.1% [-33, +5] | **-10.0% [-17, -3]** | **-14.0% [-18, -10]** | **-12.3% [-19, -6]** | **-8.3% [-15, -1]** | **-12.5% [-19, -6]** |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| autonomous | 0.0646 [0.0607, 0.0685] | 0.0420 [0.0389, 0.0451] | 0.0651 [0.0600, 0.0702] | 0.0699 [0.0648, 0.0751] | 0.0604 [0.0561, 0.0647] | 0.00 | – | 90.63 |
| dynamic | 0.0208 [0.0200, 0.0215] | 0.0190 [0.0141, 0.0239] | 0.0202 [0.0174, 0.0230] | 0.0205 [0.0189, 0.0220] | 0.0201 [0.0178, 0.0224] | 0.00 | 3555552 | 1786.85 |
| dynamic[unfreeze_base=true] | **0.0144 [0.0141, 0.0146]** | **0.0146 [0.0143, 0.0150]** | **0.0148 [0.0145, 0.0151]** | **0.0148 [0.0146, 0.0149]** | **0.0146 [0.0144, 0.0149]** | 0.00 | 3555552 | 1805.91 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0174 [0.0152, 0.0195] | 0.0168 [0.0156, 0.0180] | 0.0178 [0.0157, 0.0198] | 0.0175 [0.0163, 0.0186] | 0.0173 [0.0157, 0.0190] | 0.00 | 3555552 | 1818.94 |

Paired vs `dynamic`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| autonomous | **+211.2% [+183, +239]** | **+123.4% [+54, +193]** | **+223.6% [+154, +293]** | **+242.1% [+192, +292]** | **+201.3% [+146, +256]** |
| dynamic[unfreeze_base=true] | **-30.8% [-32, -29]** | **-22.5% [-41, -4]** | **-26.5% [-37, -16]** | **-27.9% [-34, -21]** | **-27.1% [-35, -19]** |
| dynamic[unfreeze_base=true,lr=0.0005] | **-16.4% [-24, -9]** | -11.0% [-27, +5] | **-12.0% [-19, -5]** | **-14.6% [-18, -11]** | **-13.7% [-17, -11]** |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.0840 [0.0787, 0.0892] | 0.0840 [0.0787, 0.0892] | 0.00 | – | 90.63 |
| dynamic | 0.0256 [0.0199, 0.0312] | 0.0256 [0.0199, 0.0312] | 0.00 | 3555552 | 1786.85 |
| dynamic[unfreeze_base=true] | **0.0164 [0.0161, 0.0168]** | **0.0164 [0.0161, 0.0168]** | 0.00 | 3555552 | 1805.91 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0202 [0.0167, 0.0238] | 0.0202 [0.0167, 0.0238] | 0.00 | 3555552 | 1818.94 |

Paired vs `dynamic`:

| method | drive | all |
|---|---|---|
| autonomous | **+230.4% [+144, +317]** | **+230.4% [+144, +317]** |
| dynamic[unfreeze_base=true] | **-35.4% [-49, -22]** | **-35.4% [-49, -22]** |
| dynamic[unfreeze_base=true,lr=0.0005] | **-20.6% [-41, -0]** | **-20.6% [-41, -0]** |

### fam_zero / clean

| method | zero | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.0170 [0.0163, 0.0176] | 0.0170 [0.0163, 0.0176] | 0.00 | – | 90.63 |
| dynamic | 0.0170 [0.0163, 0.0176] | 0.0170 [0.0163, 0.0176] | 0.00 | 3555552 | 1786.85 |
| dynamic[unfreeze_base=true] | **0.0142 [0.0140, 0.0145]** | **0.0142 [0.0140, 0.0145]** | 0.00 | 3555552 | 1805.91 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0163 [0.0155, 0.0171] | 0.0163 [0.0155, 0.0171] | 0.00 | 3555552 | 1818.94 |

Paired vs `dynamic`:

| method | zero | all |
|---|---|---|
| autonomous | +0.0% [+0, +0] | +0.0% [+0, +0] |
| dynamic[unfreeze_base=true] | **-16.2% [-20, -12]** | **-16.2% [-20, -12]** |
| dynamic[unfreeze_base=true,lr=0.0005] | **-4.0% [-5, -3]** | **-4.0% [-5, -3]** |

### fam_constant / clean

| method | constant | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.0778 [0.0710, 0.0846] | 0.0778 [0.0710, 0.0846] | 0.00 | – | 90.63 |
| dynamic | 0.0202 [0.0163, 0.0241] | 0.0202 [0.0163, 0.0241] | 0.00 | 3555552 | 1786.85 |
| dynamic[unfreeze_base=true] | **0.0154 [0.0148, 0.0160]** | **0.0154 [0.0148, 0.0160]** | 0.00 | 3555552 | 1805.91 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0173 [0.0166, 0.0179] | 0.0173 [0.0166, 0.0179] | 0.00 | 3555552 | 1818.94 |

Paired vs `dynamic`:

| method | constant | all |
|---|---|---|
| autonomous | **+286.9% [+181, +393]** | **+286.9% [+181, +393]** |
| dynamic[unfreeze_base=true] | **-23.5% [-40, -6]** | **-23.5% [-40, -6]** |
| dynamic[unfreeze_base=true,lr=0.0005] | **-14.3% [-27, -1]** | **-14.3% [-27, -1]** |

## iclr_rossler

phase 1: eps_enc_rel 0.0103, eps_dec 9.72, eps_pde_rms 4.07, eps_pde_ss_rms 0.907, l_dec_q95 12.7, l_dec_mean 2.89 · mean job wall 93.5 min

- state coverage id: 0.0% of eval states outside the training box, 0.7% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.7% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 1.2% beyond the training kNN q99
- state coverage fam_zero: 0.0% of eval states outside the training box, 0.5% beyond the training kNN q99
- state coverage fam_constant: 0.0% of eval states outside the training box, 0.7% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| autonomous | 0.1070 [0.0962, 0.1179] | 0.1833 [0.1788, 0.1879] | 0.2137 [0.1973, 0.2301] | 0.1473 [0.1431, 0.1514] | 0.2345 [0.2194, 0.2496] | 0.2258 [0.2133, 0.2383] | 0.1605 [0.1559, 0.1651] | 0.1886 [0.1778, 0.1995] | 0.1747 [0.1576, 0.1917] | 0.1895 [0.1768, 0.2023] | 0.1833 [0.1750, 0.1916] | 0.00 | – | 90.41 |
| dynamic | 0.1019 [0.0868, 0.1169] | 0.0924 [0.0898, 0.0950] | 0.1057 [0.1006, 0.1108] | 0.1033 [0.0950, 0.1117] | 0.1129 [0.1068, 0.1190] | 0.1025 [0.0992, 0.1059] | 0.1031 [0.0926, 0.1137] | 0.0964 [0.0927, 0.1002] | 0.1096 [0.1023, 0.1169] | 0.1087 [0.1059, 0.1114] | 0.1038 [0.1000, 0.1076] | 0.00 | 14234880 | 7614.14 |
| dynamic[unfreeze_base=true] | **0.0841 [0.0787, 0.0896]** | **0.0783 [0.0752, 0.0813]** | **0.0979 [0.0933, 0.1025]** | **0.0876 [0.0851, 0.0900]** | **0.0996 [0.0890, 0.1103]** | **0.0857 [0.0845, 0.0868]** | **0.0868 [0.0832, 0.0905]** | **0.0865 [0.0843, 0.0887]** | **0.0894 [0.0870, 0.0918]** | **0.0912 [0.0895, 0.0928]** | **0.0888 [0.0860, 0.0915]** | 0.00 | 14234880 | 7294.10 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0851 [0.0815, 0.0887] | 0.0799 [0.0777, 0.0822] | 0.0984 [0.0950, 0.1017] | 0.0911 [0.0885, 0.0936] | 0.1001 [0.0945, 0.1056] | 0.0899 [0.0876, 0.0923] | 0.0882 [0.0858, 0.0905] | 0.0882 [0.0872, 0.0891] | 0.0925 [0.0904, 0.0946] | 0.0947 [0.0914, 0.0980] | 0.0909 [0.0892, 0.0925] | 0.00 | 14234880 | 8475.36 |

Paired vs `dynamic`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| autonomous | +5.1% [-0, +10] | **+98.4% [+93, +103]** | **+102.2% [+91, +113]** | **+42.6% [+27, +58]** | **+107.8% [+86, +130]** | **+120.3% [+104, +137]** | **+55.7% [+42, +69]** | **+95.6% [+77, +114]** | **+59.4% [+42, +77]** | **+74.4% [+66, +82]** | **+76.6% [+63, +90]** |
| dynamic[unfreeze_base=true] | **-17.2% [-33, -1]** | **-15.3% [-21, -10]** | **-7.4% [-11, -4]** | **-15.2% [-20, -11]** | **-11.8% [-18, -5]** | **-16.4% [-20, -13]** | **-15.7% [-26, -5]** | **-10.3% [-16, -5]** | **-18.4% [-24, -12]** | **-16.1% [-19, -14]** | **-14.5% [-19, -10]** |
| dynamic[unfreeze_base=true,lr=0.0005] | **-16.2% [-30, -2]** | **-13.5% [-18, -9]** | **-6.9% [-12, -2]** | **-11.8% [-20, -3]** | **-11.4% [-14, -9]** | **-12.3% [-17, -7]** | **-14.4% [-25, -4]** | **-8.6% [-13, -4]** | **-15.6% [-22, -9]** | **-12.9% [-14, -12]** | **-12.5% [-17, -8]** |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| autonomous | 0.1658 [0.1535, 0.1781] | 0.1519 [0.1464, 0.1575] | 0.1795 [0.1675, 0.1916] | 0.1499 [0.1456, 0.1543] | 0.1618 [0.1538, 0.1698] | 0.00 | – | 90.41 |
| dynamic | 0.1033 [0.0991, 0.1074] | 0.1147 [0.1095, 0.1198] | 0.1063 [0.1026, 0.1100] | 0.1122 [0.1061, 0.1184] | 0.1091 [0.1055, 0.1127] | 0.00 | 14234880 | 7614.14 |
| dynamic[unfreeze_base=true] | **0.0830 [0.0823, 0.0836]** | **0.1029 [0.1015, 0.1043]** | 0.0917 [0.0821, 0.1014] | **0.0946 [0.0937, 0.0954]** | **0.0930 [0.0901, 0.0959]** | 0.00 | 14234880 | 7294.10 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0872 [0.0858, 0.0887] | 0.1046 [0.1034, 0.1058] | **0.0915 [0.0864, 0.0967]** | 0.0980 [0.0962, 0.0998] | 0.0953 [0.0929, 0.0977] | 0.00 | 14234880 | 8475.36 |

Paired vs `dynamic`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| autonomous | **+60.6% [+47, +74]** | **+32.5% [+26, +39]** | **+69.0% [+52, +86]** | **+33.6% [+29, +38]** | **+48.3% [+39, +58]** |
| dynamic[unfreeze_base=true] | **-19.6% [-23, -16]** | **-10.2% [-15, -5]** | **-13.7% [-21, -7]** | **-15.7% [-20, -12]** | **-14.7% [-20, -10]** |
| dynamic[unfreeze_base=true,lr=0.0005] | **-15.5% [-20, -11]** | **-8.7% [-13, -4]** | **-13.9% [-16, -12]** | **-12.7% [-18, -7]** | **-12.6% [-16, -9]** |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.2265 [0.2041, 0.2488] | 0.2265 [0.2041, 0.2488] | 0.00 | – | 90.41 |
| dynamic | 0.1144 [0.1124, 0.1164] | 0.1144 [0.1124, 0.1164] | 0.00 | 14234880 | 7614.14 |
| dynamic[unfreeze_base=true] | **0.0891 [0.0810, 0.0973]** | **0.0891 [0.0810, 0.0973]** | 0.00 | 14234880 | 7294.10 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0913 [0.0895, 0.0931] | 0.0913 [0.0895, 0.0931] | 0.00 | 14234880 | 8475.36 |

Paired vs `dynamic`:

| method | drive | all |
|---|---|---|
| autonomous | **+98.0% [+79, +116]** | **+98.0% [+79, +116]** |
| dynamic[unfreeze_base=true] | **-22.1% [-28, -16]** | **-22.1% [-28, -16]** |
| dynamic[unfreeze_base=true,lr=0.0005] | **-20.2% [-22, -18]** | **-20.2% [-22, -18]** |

### fam_zero / clean

| method | zero | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.1056 [0.0938, 0.1174] | 0.1056 [0.0938, 0.1174] | 0.00 | – | 90.41 |
| dynamic | 0.1056 [0.0938, 0.1174] | 0.1056 [0.0938, 0.1174] | 0.00 | 14234880 | 7614.14 |
| dynamic[unfreeze_base=true] | **0.0892 [0.0877, 0.0907]** | **0.0892 [0.0877, 0.0907]** | 0.00 | 14234880 | 7294.10 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0916 [0.0911, 0.0922] | 0.0916 [0.0911, 0.0922] | 0.00 | 14234880 | 8475.36 |

Paired vs `dynamic`:

| method | zero | all |
|---|---|---|
| autonomous | +0.0% [+0, +0] | +0.0% [+0, +0] |
| dynamic[unfreeze_base=true] | **-15.4% [-24, -7]** | **-15.4% [-24, -7]** |
| dynamic[unfreeze_base=true,lr=0.0005] | **-13.1% [-23, -3]** | **-13.1% [-23, -3]** |

### fam_constant / clean

| method | constant | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.1844 [0.1752, 0.1937] | 0.1844 [0.1752, 0.1937] | 0.00 | – | 90.41 |
| dynamic | 0.0975 [0.0930, 0.1020] | 0.0975 [0.0930, 0.1020] | 0.00 | 14234880 | 7614.14 |
| dynamic[unfreeze_base=true] | **0.0853 [0.0842, 0.0863]** | **0.0853 [0.0842, 0.0863]** | 0.00 | 14234880 | 7294.10 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0884 [0.0875, 0.0892] | 0.0884 [0.0875, 0.0892] | 0.00 | 14234880 | 8475.36 |

Paired vs `dynamic`:

| method | constant | all |
|---|---|---|
| autonomous | **+89.2% [+73, +105]** | **+89.2% [+73, +105]** |
| dynamic[unfreeze_base=true] | **-12.5% [-16, -9]** | **-12.5% [-16, -9]** |
| dynamic[unfreeze_base=true,lr=0.0005] | **-9.4% [-14, -5]** | **-9.4% [-14, -5]** |

## iclr_lorenz

phase 1: eps_enc_rel 0.177, eps_dec 4.52, eps_pde_rms 78.3, eps_pde_ss_rms 8.71, l_dec_q95 3.11, l_dec_mean 1.33 · mean job wall 92.9 min

- state coverage id: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.3% beyond the training kNN q99
- state coverage fam_zero: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage fam_constant: 0.0% of eval states outside the training box, 0.3% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| autonomous | 0.0169 [0.0146, 0.0192] | 0.0171 [0.0148, 0.0195] | 0.0176 [0.0150, 0.0203] | 0.0180 [0.0152, 0.0208] | 0.0162 [0.0139, 0.0185] | 0.0169 [0.0142, 0.0197] | 0.0172 [0.0146, 0.0198] | 0.0195 [0.0164, 0.0226] | 0.0153 [0.0137, 0.0170] | 0.0170 [0.0147, 0.0192] | 0.0171 [0.0147, 0.0196] | 0.00 | – | 89.64 |
| dynamic | 0.0168 [0.0145, 0.0191] | 0.0170 [0.0145, 0.0194] | 0.0175 [0.0147, 0.0202] | 0.0177 [0.0151, 0.0204] | 0.0161 [0.0138, 0.0185] | 0.0166 [0.0139, 0.0193] | 0.0171 [0.0144, 0.0199] | 0.0191 [0.0160, 0.0222] | 0.0151 [0.0133, 0.0169] | 0.0168 [0.0143, 0.0192] | 0.0169 [0.0144, 0.0194] | 0.00 | 14234880 | 6466.02 |
| dynamic[unfreeze_base=true] | **0.0153 [0.0119, 0.0186]** | **0.0154 [0.0121, 0.0188]** | **0.0156 [0.0118, 0.0195]** | **0.0160 [0.0120, 0.0200]** | **0.0146 [0.0115, 0.0177]** | **0.0148 [0.0112, 0.0183]** | **0.0155 [0.0121, 0.0190]** | **0.0174 [0.0132, 0.0216]** | **0.0138 [0.0115, 0.0162]** | **0.0151 [0.0119, 0.0183]** | **0.0153 [0.0119, 0.0187]** | 0.00 | 14234880 | 6396.59 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0161 [0.0137, 0.0185] | 0.0162 [0.0136, 0.0188] | 0.0166 [0.0137, 0.0195] | 0.0169 [0.0141, 0.0197] | 0.0154 [0.0129, 0.0178] | 0.0157 [0.0128, 0.0186] | 0.0163 [0.0135, 0.0191] | 0.0183 [0.0150, 0.0215] | 0.0145 [0.0126, 0.0163] | 0.0160 [0.0135, 0.0184] | 0.0161 [0.0135, 0.0188] | 0.00 | 14234880 | 6298.00 |

Paired vs `dynamic`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| autonomous | **+0.8% [+1, +1]** | **+0.9% [+0, +2]** | **+1.0% [+0, +2]** | **+1.5% [+1, +2]** | **+0.7% [+0, +1]** | **+1.9% [+2, +2]** | +0.7% [-1, +2] | **+2.1% [+2, +3]** | **+1.4% [+0, +3]** | +1.1% [-0, +2] | **+1.2% [+1, +2]** |
| dynamic[unfreeze_base=true] | -9.1% [-21, +3] | -9.1% [-20, +2] | -10.5% [-23, +2] | -10.0% [-24, +4] | -9.3% [-19, +1] | -11.1% [-22, +0] | -9.2% [-20, +2] | -9.0% [-22, +3] | **-8.4% [-15, -2]** | -9.8% [-21, +1] | -9.5% [-21, +1] |
| dynamic[unfreeze_base=true,lr=0.0005] | **-4.2% [-7, -1]** | **-4.5% [-8, -1]** | **-5.0% [-9, -1]** | **-4.7% [-9, -0]** | **-4.6% [-7, -2]** | **-5.4% [-10, -1]** | **-4.6% [-8, -1]** | **-4.4% [-8, -1]** | **-4.3% [-6, -2]** | **-4.9% [-8, -2]** | **-4.7% [-8, -1]** |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| autonomous | 0.0173 [0.0146, 0.0200] | 0.0165 [0.0141, 0.0188] | 0.0182 [0.0153, 0.0210] | 0.0172 [0.0148, 0.0197] | 0.0173 [0.0147, 0.0199] | 0.00 | – | 89.64 |
| dynamic | 0.0171 [0.0143, 0.0198] | 0.0163 [0.0139, 0.0186] | 0.0179 [0.0150, 0.0208] | 0.0170 [0.0145, 0.0196] | 0.0171 [0.0144, 0.0197] | 0.00 | 14234880 | 6466.02 |
| dynamic[unfreeze_base=true] | **0.0154 [0.0118, 0.0190]** | **0.0148 [0.0116, 0.0180]** | **0.0162 [0.0122, 0.0201]** | **0.0155 [0.0123, 0.0187]** | **0.0155 [0.0120, 0.0190]** | 0.00 | 14234880 | 6396.59 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0162 [0.0134, 0.0191] | 0.0155 [0.0131, 0.0180] | 0.0170 [0.0140, 0.0200] | 0.0163 [0.0137, 0.0188] | 0.0163 [0.0136, 0.0190] | 0.00 | 14234880 | 6298.00 |

Paired vs `dynamic`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| autonomous | **+1.5% [+1, +2]** | **+1.4% [+1, +2]** | **+1.6% [+1, +2]** | **+1.3% [+1, +2]** | **+1.4% [+1, +2]** |
| dynamic[unfreeze_base=true] | -9.7% [-20, +1] | -8.8% [-19, +1] | -9.5% [-22, +3] | -9.0% [-19, +1] | -9.3% [-20, +2] |
| dynamic[unfreeze_base=true,lr=0.0005] | **-4.9% [-8, -1]** | **-4.4% [-8, -1]** | **-4.7% [-9, -1]** | **-4.4% [-7, -2]** | **-4.6% [-8, -1]** |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.0176 [0.0150, 0.0201] | 0.0176 [0.0150, 0.0201] | 0.00 | – | 89.64 |
| dynamic | 0.0172 [0.0147, 0.0198] | 0.0172 [0.0147, 0.0198] | 0.00 | 14234880 | 6466.02 |
| dynamic[unfreeze_base=true] | **0.0157 [0.0122, 0.0191]** | **0.0157 [0.0122, 0.0191]** | 0.00 | 14234880 | 6396.59 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0164 [0.0138, 0.0191] | 0.0164 [0.0138, 0.0191] | 0.00 | 14234880 | 6298.00 |

Paired vs `dynamic`:

| method | drive | all |
|---|---|---|
| autonomous | **+1.9% [+1, +2]** | **+1.9% [+1, +2]** |
| dynamic[unfreeze_base=true] | -9.2% [-20, +2] | -9.2% [-20, +2] |
| dynamic[unfreeze_base=true,lr=0.0005] | **-4.6% [-8, -1]** | **-4.6% [-8, -1]** |

### fam_zero / clean

| method | zero | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.0170 [0.0143, 0.0196] | 0.0170 [0.0143, 0.0196] | 0.00 | – | 89.64 |
| dynamic | 0.0170 [0.0143, 0.0196] | 0.0170 [0.0143, 0.0196] | 0.00 | 14234880 | 6466.02 |
| dynamic[unfreeze_base=true] | **0.0154 [0.0119, 0.0189]** | **0.0154 [0.0119, 0.0189]** | 0.00 | 14234880 | 6396.59 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0161 [0.0134, 0.0189] | 0.0161 [0.0134, 0.0189] | 0.00 | 14234880 | 6298.00 |

Paired vs `dynamic`:

| method | zero | all |
|---|---|---|
| autonomous | +0.0% [+0, +0] | +0.0% [+0, +0] |
| dynamic[unfreeze_base=true] | -9.4% [-21, +2] | -9.4% [-21, +2] |
| dynamic[unfreeze_base=true,lr=0.0005] | **-4.8% [-8, -1]** | **-4.8% [-8, -1]** |

### fam_constant / clean

| method | constant | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.0172 [0.0147, 0.0196] | 0.0172 [0.0147, 0.0196] | 0.00 | – | 89.64 |
| dynamic | 0.0168 [0.0143, 0.0193] | 0.0168 [0.0143, 0.0193] | 0.00 | 14234880 | 6466.02 |
| dynamic[unfreeze_base=true] | **0.0153 [0.0121, 0.0185]** | **0.0153 [0.0121, 0.0185]** | 0.00 | 14234880 | 6396.59 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0161 [0.0135, 0.0186] | 0.0161 [0.0135, 0.0186] | 0.00 | 14234880 | 6298.00 |

Paired vs `dynamic`:

| method | constant | all |
|---|---|---|
| autonomous | **+2.2% [+1, +3]** | **+2.2% [+1, +3]** |
| dynamic[unfreeze_base=true] | -9.0% [-19, +1] | -9.0% [-19, +1] |
| dynamic[unfreeze_base=true,lr=0.0005] | **-4.5% [-8, -1]** | **-4.5% [-8, -1]** |

## Deployment scorecard

One row per method. nrmse columns: clean condition, mean over episodes and seeds; unmeasured = mean NRMSE over the unmeasured states; worst sensor fault = max over the sensor conditions present (named); p99 over episodes; div. rate = max over every tier/condition block (named); forget prior = median time until the poor_init estimate merges with the same observer's nominal-prior estimate (paired); conv. = median time after which the poor_init error stays below 2x its settled median; recovery = median time to return within 1.5x the pre-outage error after burst_outage; drift = NRMSE(last 20%) / NRMSE(20% after settling) on `long` episodes; alarm AUROC = output-residual alarm detecting state error > 3x median (metrics.py). Tier: sequence_ood unless named in the cell.

### iclr_duffing

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| autonomous | 0.2079 | 0.1828 | 0.1383 | 0.1837 | 0.2212 | 0.00 (shape_ood/clean) | 0.232 | 88.6 |
| dynamic | 0.1340 | 0.1174 | 0.0878 | 0.1155 | 0.2039 | 0.00 (shape_ood/clean) | 0.527 | 1.8e+03 |
| dynamic[unfreeze_base=true] | 0.1231 | 0.1072 | 0.0912 | 0.1216 | 0.2719 | 0.00 (shape_ood/clean) | 0.526 | 1.81e+03 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.1335 | 0.1170 | 0.0912 | 0.1262 | 0.1789 | 0.00 (shape_ood/clean) | 0.515 | 1.8e+03 |

### iclr_vdp

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| autonomous | 0.0637 | 0.0604 | 0.0840 | 0.1180 | 0.1108 | 0.00 (shape_ood/clean) | 0.622 | 91.8 |
| dynamic | 0.0198 | 0.0201 | 0.0256 | 0.0346 | 0.0348 | 0.00 (shape_ood/clean) | 0.512 | 1.8e+03 |
| dynamic[unfreeze_base=true] | 0.0149 | 0.0146 | 0.0164 | 0.0212 | 0.0195 | 0.00 (shape_ood/clean) | 0.541 | 1.8e+03 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0173 | 0.0173 | 0.0202 | 0.0268 | 0.0240 | 0.00 (shape_ood/clean) | 0.521 | 1.81e+03 |

### iclr_rossler

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| autonomous | 0.1833 | 0.1618 | 0.2265 | 0.2292 | 0.3199 | 0.00 (shape_ood/clean) | 0.561 | 92.7 |
| dynamic | 0.1038 | 0.1091 | 0.1144 | 0.1167 | 0.1991 | 0.00 (shape_ood/clean) | 0.544 | 6.94e+03 |
| dynamic[unfreeze_base=true] | 0.0888 | 0.0930 | 0.0891 | 0.0892 | 0.1467 | 0.00 (shape_ood/clean) | 0.505 | 6.8e+03 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0909 | 0.0953 | 0.0913 | 0.0921 | 0.1505 | 0.00 (shape_ood/clean) | 0.509 | 7.19e+03 |

### iclr_lorenz

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| autonomous | 0.0171 | 0.0173 | 0.0176 | 0.0168 | 0.0274 | 0.00 (shape_ood/clean) | 0.528 | 90.5 |
| dynamic | 0.0169 | 0.0171 | 0.0172 | 0.0165 | 0.0271 | 0.00 (shape_ood/clean) | 0.535 | 6.89e+03 |
| dynamic[unfreeze_base=true] | 0.0153 | 0.0155 | 0.0157 | 0.0152 | 0.0245 | 0.00 (shape_ood/clean) | 0.550 | 6.81e+03 |
| dynamic[unfreeze_base=true,lr=0.0005] | 0.0161 | 0.0163 | 0.0164 | 0.0158 | 0.0261 | 0.00 (shape_ood/clean) | 0.532 | 6.52e+03 |
