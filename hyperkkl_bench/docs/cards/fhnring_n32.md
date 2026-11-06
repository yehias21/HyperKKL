# Plant card: `fhnring_n32`

- role: **scaling**
- n_x = 32, n_y = 8, n_u = 1; dt = 0.05 s, N = 2000, T = 100 s, t_ref = 1.5 s
- measured: v0, v2, v4, v6, v8, v10, v12, v14
- unmeasured: w0, v1, w1, w2, v3, w3, w4, v5, w5, w6, v7, w7, w8, v9, w9, w10, v11, w11, w12, v13, w13, w14, v15, w15
- partial model misses: the excitatory chemical synapse g_syn (E_syn - v_i) sum_j a_ij sig((v_j - th)/s) between ring neighbours (model: gap-junction coupling only)
- input box: u in [-1.0, 1.0] (actuator: saturating)
- information group: None

## Difficulty axes

| axis | value | provenance |
|---|---|---|

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood |
|---|---|---|---|---|---|---|
| white | 2400 | 64 | 1600 | 64 | 64 | 64 |
| mismatch30 | 2400 | 64 | 1600 | 64 | 64 | 64 |
| realistic | 2400 | 64 | 1600 | 64 | 64 | 64 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
