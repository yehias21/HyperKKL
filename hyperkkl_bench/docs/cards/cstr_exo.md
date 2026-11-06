# Plant card: `cstr_exo`

- role: **deployment**
- n_x = 3, n_y = 2, n_u = 1; dt = 0.166667 s, N = 2880, T = 480 s, t_ref = 20 s
- measured: T (K), T_j (K)
- unmeasured: C_A (mol/L)
- partial model misses: the side reaction A -> D (k2, E2, dH2): the practitioner's model has the main reaction A -> B only
- input box: u in [-0.5, 0.5] (actuator: affine)
- information group: None

## Difficulty axes

| axis | value | provenance |
|---|---|---|

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood |
|---|---|---|---|---|---|---|
| white | 27 | 6 | 6 | 64 | 64 | 64 |
| mismatch30 | 27 | 6 | 6 | 64 | 64 | 64 |
| realistic | 27 | 6 | 6 | 64 | 64 | 64 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
