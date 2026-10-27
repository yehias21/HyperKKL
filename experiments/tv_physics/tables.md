### Test 1 (confirmatory): TV loss ON (chosen lambda~) vs OFF

| tier | battery/partial | bioreactor/partial | lorenz/partial | quad_tank/partial | vdp/partial | battery/mm30 | bioreactor/mm30 | lorenz/mm30 | quad_tank/mm30 | vdp/mm30 |
|---|---|---|---|---|---|---|---|---|---|---|
| sequence_ood | -7% [-30, +25] | +3% [-8, +16] | -0% [-2, +1] | -15% [-57, +71] | -0% [-2, +1] | -2% [-8, +3] | +2% [-10, +15] | -1% [-7, +5] | +8% [-39, +90] | -0% [-2, +2] |
| shape_ood | -3% [-19, +16] | *+2% [+0, +3]* | -0% [-2, +1] | -17% [-63, +85] | -0% [-1, +1] | -4% [-11, +4] | -1% [-2, +0] | -1% [-5, +4] | +7% [-37, +84] | +0% [-1, +1] |
| id | +8% [-17, +41] | +1% [-2, +4] | -0% [-2, +1] | -18% [-64, +87] | -0% [-0, +0] | -2% [-4, +1] | -0% [-1, +1] | -1% [-5, +4] | +9% [-41, +103] | -0% [-1, +1] |
| Holm p (seq) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| arm on (seq NRMSE on/off, seed means) | TVr0.1 0.0243/0.0260 | TVr0.1 0.0142/0.0137 | TVr1 0.0078/0.0078 | TVr0.1 0.0234/0.0286 | TVr0.1 0.0066/0.0067 | TVr0.1 0.0067/0.0069 | TVr0.1 0.0112/0.0110 | TVr0.1 0.0039/0.0040 | TVr1 0.0254/0.0235 | TVr0.1 0.0070/0.0070 |

Effect = NRMSE_on / NRMSE_off - 1, paired t 95% CI on the log ratio over seeds (df = 2); **bold** = on significantly better, *italic* = significantly worse.

verdict: {"C3-partial": {"better": 0, "worse": 0, "helps": false, "complete": true, "holm_better": 0}, "C3-mismatch30": {"better": 0, "worse": 0, "helps": false, "complete": true, "holm_better": 0}, "overall": "does not help"}

### Test 2 (confirmatory): phase-1 PDE ON (A4+D3) vs OFF (P1OFF)

| tier | battery/partial | bioreactor/partial | lorenz/partial | quad_tank/partial | vdp/partial | battery/mm30 | bioreactor/mm30 | lorenz/mm30 | quad_tank/mm30 | vdp/mm30 |
|---|---|---|---|---|---|---|---|---|---|---|
| sequence_ood | +15% [-1, +34] | -4% [-10, +2] | *+7% [+5, +8]* | -18% [-62, +77] | +0% [-2, +2] | -2% [-30, +35] | +1% [-14, +19] | +1% [-4, +7] | +5% [-30, +57] | -1% [-2, +0] |
| shape_ood | -14% [-32, +8] | -1% [-10, +8] | *+5% [+5, +6]* | -17% [-60, +75] | -0% [-1, +0] | **-12% [-21, -2]** | **-1% [-2, -1]** | +1% [-3, +6] | +4% [-15, +27] | -0% [-1, +1] |
| id | **-13% [-19, -6]** | +1% [-3, +5] | *+6% [+4, +7]* | -19% [-65, +85] | -0% [-1, +0] | -5% [-29, +27] | -0% [-3, +2] | +1% [-3, +6] | +3% [-13, +24] | -1% [-2, +1] |
| Holm p (seq) | 0.48 | 0.80 | 0.02 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.48 |
| arm on (seq NRMSE on/off, seed means) | OFF 0.0260/0.0225 | OFF 0.0137/0.0142 | OFF 0.0078/0.0073 | OFF 0.0286/0.0346 | OFF 0.0067/0.0066 | OFF 0.0069/0.0071 | OFF 0.0110/0.0109 | OFF 0.0040/0.0039 | OFF 0.0235/0.0224 | OFF 0.0070/0.0071 |

Effect = NRMSE_on / NRMSE_off - 1, paired t 95% CI on the log ratio over seeds (df = 2); **bold** = on significantly better, *italic* = significantly worse.

verdict: {"C3-partial": {"better": 0, "worse": 1, "helps": false, "complete": true, "holm_better": 0}, "C3-mismatch30": {"better": 0, "worse": 0, "helps": false, "complete": true, "holm_better": 0}, "overall": "does not help"}

### Exploratory: TV with the ORACLE (true) f, lambda~ = 1, vs OFF

| tier | battery/partial | bioreactor/partial | lorenz/partial | quad_tank/partial | vdp/partial | battery/mm30 | bioreactor/mm30 | lorenz/mm30 | quad_tank/mm30 | vdp/mm30 |
|---|---|---|---|---|---|---|---|---|---|---|
| sequence_ood | -3% [-22, +21] | +5% [-9, +20] | -1% [-2, +1] | +1% [-9, +13] | +0% [-3, +3] | -2% [-11, +8] | +0% [-7, +8] | -1% [-7, +4] | +2% [-24, +38] | +0% [-1, +1] |
| shape_ood | -0% [-23, +28] | -0% [-2, +2] | -0% [-2, +1] | +3% [-8, +16] | +0% [-1, +1] | -3% [-16, +12] | -1% [-4, +3] | -1% [-5, +3] | -1% [-21, +25] | +0% [-0, +1] |
| id | -3% [-11, +6] | +0% [-3, +4] | -0% [-2, +1] | +4% [-10, +20] | -0% [-1, +1] | -0% [-6, +6] | +0% [-1, +1] | -1% [-5, +3] | -0% [-23, +29] | *+0% [+0, +0]* |
| Holm p (seq) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| arm on (seq NRMSE on/off, seed means) | TOr1 0.0253/0.0260 | TOr1 0.0144/0.0137 | TOr1 0.0077/0.0078 | TOr1 0.0292/0.0286 | TOr1 0.0067/0.0067 | TOr1 0.0067/0.0069 | TOr1 0.0110/0.0110 | TOr1 0.0039/0.0040 | TOr1 0.0240/0.0235 | TOr1 0.0070/0.0070 |

Effect = NRMSE_on / NRMSE_off - 1, paired t 95% CI on the log ratio over seeds (df = 2); **bold** = on significantly better, *italic* = significantly worse.

verdict (same rule, exploratory): {"C3-partial": {"better": 0, "worse": 0, "helps": false, "complete": true, "holm_better": 0}, "C3-mismatch30": {"better": 0, "worse": 0, "helps": false, "complete": true, "holm_better": 0}, "overall": "does not help"}

### Exploratory: oracle-f TV vs modeller-f TV at lambda~ = 1

| tier | battery/partial | bioreactor/partial | lorenz/partial | quad_tank/partial | vdp/partial | battery/mm30 | bioreactor/mm30 | lorenz/mm30 | quad_tank/mm30 | vdp/mm30 |
|---|---|---|---|---|---|---|---|---|---|---|
| sequence_ood | n/a | n/a | -0% [-1, +1] | n/a | n/a | n/a | n/a | n/a | -5% [-30, +28] | n/a |
| shape_ood | n/a | n/a | -0% [-0, +0] | n/a | n/a | n/a | n/a | n/a | -8% [-34, +29] | n/a |
| id | n/a | n/a | -0% [-1, +0] | n/a | n/a | n/a | n/a | n/a | -9% [-38, +34] | n/a |
| Holm p (seq) | n/a | n/a | 1.00 | n/a | n/a | n/a | n/a | n/a | 1.00 | n/a |
| arm on (seq NRMSE on/off, seed means) | TOr1 0.0225/0.0222 | TOr1 0.0126/0.0123 | TOr1 0.0077/0.0078 | TOr1 0.0432/0.0242 | TOr1 0.0068/0.0067 | TOr1 0.0067/0.0078 | TOr1 0.0108/0.0112 | TOr1 0.0039/0.0045 | TOr1 0.0240/0.0254 | TOr1 0.0071/0.0072 |

Effect = NRMSE_on / NRMSE_off - 1, paired t 95% CI on the log ratio over seeds (df = 2); **bold** = on significantly better, *italic* = significantly worse.


### Supplementary: TVr1 (fixed lambda~ = 1) vs OFF

| tier | battery/partial | bioreactor/partial | lorenz/partial | quad_tank/partial | vdp/partial | battery/mm30 | bioreactor/mm30 | lorenz/mm30 | quad_tank/mm30 | vdp/mm30 |
|---|---|---|---|---|---|---|---|---|---|---|
| sequence_ood | n/a | n/a | -0% [-2, +1] | n/a | n/a | n/a | n/a | n/a | +8% [-39, +90] | n/a |
| shape_ood | n/a | n/a | -0% [-2, +1] | n/a | n/a | n/a | n/a | n/a | +7% [-37, +84] | n/a |
| id | n/a | n/a | -0% [-2, +1] | n/a | n/a | n/a | n/a | n/a | +9% [-41, +103] | n/a |
| Holm p (seq) | n/a | n/a | 0.81 | n/a | n/a | n/a | n/a | n/a | 0.81 | n/a |
| arm on (seq NRMSE on/off, seed means) | TVr1 0.0222/0.0255 | TVr1 0.0123/0.0124 | TVr1 0.0078/0.0078 | TVr1 0.0242/0.0406 | TVr1 0.0067/0.0067 | TVr1 0.0078/0.0068 | TVr1 0.0112/0.0106 | TVr1 0.0045/0.0041 | TVr1 0.0254/0.0235 | TVr1 0.0072/0.0070 |

Effect = NRMSE_on / NRMSE_off - 1, paired t 95% CI on the log ratio over seeds (df = 2); **bold** = on significantly better, *italic* = significantly worse.


### lambda~ selection (seed-0 validation NRMSE)

| cell | lambda~=0.1 | lambda~=1 | lambda~=10 | chosen |
|---|---|---|---|---|
| C3-partial/battery | 0.01436 | 0.01479 | 0.03073 | 0.1 |
| C3-partial/bioreactor | 0.00662 | 0.00665 | 0.00662 | 0.1 |
| C3-partial/lorenz | 0.00763 | 0.00763 | 0.00774 | 1 |
| C3-partial/quad_tank | 0.01779 | 0.01789 | 0.01783 | 0.1 |
| C3-partial/vdp | 0.00595 | 0.00596 | 0.00677 | 0.1 |
| C3-mismatch30/battery | 0.00603 | 0.00662 | 0.01313 | 0.1 |
| C3-mismatch30/bioreactor | 0.00578 | 0.00580 | 0.00590 | 0.1 |
| C3-mismatch30/lorenz | 0.00377 | 0.00417 | 0.00384 | 0.1 |
| C3-mismatch30/quad_tank | 0.01452 | 0.01414 | 0.01540 | 1 |
| C3-mismatch30/vdp | 0.00639 | 0.00648 | 0.00898 | 0.1 |

### TV residual diagnostic (validation, seed means; units sigma_x * alpha)

| cell | floor (perfect decoder, modeller f) | OFF residual | TV chosen residual | TOr1 residual (vs true f) |
|---|---|---|---|---|
| battery/partial | 1.35 | 1.36 | 1.34 | 0.165 |
| bioreactor/partial | 12 | 12 | 12 | 0.0248 |
| lorenz/partial | 7.78 | 7.79 | 7.78 | 0.279 |
| quad_tank/partial | 0.00802 | 0.361 | 0.135 | 0.136 |
| vdp/partial | 0.374 | 0.376 | 0.376 | 0.0275 |
| battery/mm30 | 0.653 | 0.656 | 0.653 | 0.0632 |
| bioreactor/mm30 | 4.77 | 4.77 | 4.77 | 0.0284 |
| lorenz/mm30 | 2.07 | 2.07 | 2.07 | 0.0464 |
| quad_tank/mm30 | 1.9 | 1.91 | 1.9 | 0.185 |
| vdp/mm30 | 0.15 | 0.155 | 0.153 | 0.0202 |
