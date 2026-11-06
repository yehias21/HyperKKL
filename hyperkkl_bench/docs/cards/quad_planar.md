# Plant card: `quad_planar`

- role: **deployment**
- n_x = 7, n_y = 4, n_u = 1; dt = 0.01 s, N = 2000, T = 20 s, t_ref = 1 s
- measured: px (m), pz (m), q (rad/s), a_m (m/s^2)
- unmeasured: vx (m/s), vz (m/s), theta (rad)
- partial model misses: rotor drag (kd_x, kd_z): the practitioner's rigid-body + thrust model has no aerodynamic drag, so its accelerometer channel predicts a_bx = 0
- input box: u in [-3.0, 3.0] (actuator: affine)
- information group: None

## Difficulty axes

| axis | value | provenance |
|---|---|---|

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood |
|---|---|---|---|---|---|---|
| white | 60 | 12 | 30 | 64 | 64 | 64 |
| mismatch30 | 60 | 12 | 30 | 64 | 64 | 64 |
| realistic | 60 | 12 | 30 | 64 | 64 | 64 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
