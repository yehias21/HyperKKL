Coverage (settled eval steps, mean over seeds 0-2; errors = RMS |xhat - x| / x_std, unmeasured = mean over unmeasured states):

| plant | model | tier | joint outside min/max | unm err inside | unm err outside | ratio | rule | outside 1-99% band (secondary) | unm err in band | unm err out of band | x3 in / out band | x4 in / out band |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cartpole | A2 | id | 0.0% | 0.709 | nan | nan | no | 0.0% | 0.709 | nan | 0.661 / nan | 1.071 / nan |
| cartpole | A2 | shape_ood | 0.0% | 0.623 | nan | nan | no | 0.0% | 0.623 | nan | 0.567 / nan | 1.016 / nan |
| cartpole | A2 | sequence_ood | 0.0% | 1.171 | nan | nan | no | 0.0% | 1.171 | nan | 0.963 / nan | 2.112 / nan |
| cartpole | A4 | id | 0.0% | 0.372 | nan | nan | no | 0.0% | 0.372 | nan | 0.261 / nan | 0.571 / nan |
| cartpole | A4 | shape_ood | 0.0% | 0.280 | nan | nan | no | 0.0% | 0.280 | nan | 0.213 / nan | 0.420 / nan |
| cartpole | A4 | sequence_ood | 0.0% | 0.644 | nan | nan | no | 0.0% | 0.644 | nan | 0.444 / nan | 1.114 / nan |
| cartpole | lora | id | 0.0% | 0.466 | nan | nan | no | 0.0% | 0.466 | nan | 0.340 / nan | 0.732 / nan |
| cartpole | lora | shape_ood | 0.0% | 0.367 | nan | nan | no | 0.0% | 0.367 | nan | 0.273 / nan | 0.590 / nan |
| cartpole | lora | sequence_ood | 0.0% | 0.650 | nan | nan | no | 0.0% | 0.650 | nan | 0.468 / nan | 1.125 / nan |
| cartpole | greybox_ukf | id | 0.0% | 0.053 | nan | nan | no | 0.0% | 0.053 | nan | 0.006 / nan | 0.085 / nan |
| cartpole | greybox_ukf | shape_ood | 0.0% | 0.049 | nan | nan | no | 0.0% | 0.049 | nan | 0.006 / nan | 0.079 / nan |
| cartpole | greybox_ukf | sequence_ood | 0.0% | 0.072 | nan | nan | no | 0.0% | 0.072 | nan | 0.008 / nan | 0.126 / nan |
| highway_traffic | lora | id | 0.0% | 0.212 | nan | nan | no | 0.1% | 0.212 | 0.511 | - | - |
| highway_traffic | lora | shape_ood | 0.0% | 0.208 | nan | nan | no | 0.0% | 0.208 | nan | - | - |
| highway_traffic | lora | sequence_ood | 0.0% | 0.219 | nan | nan | no | 0.0% | 0.219 | nan | - | - |
| quad_tank | A2 | id | 0.0% | 0.038 | nan | nan | no | 7.4% | 0.039 | 0.035 | - | - |
| quad_tank | A2 | shape_ood | 0.0% | 0.038 | nan | nan | no | 0.0% | 0.038 | nan | - | - |
| quad_tank | A2 | sequence_ood | 0.0% | 0.047 | nan | nan | no | 0.0% | 0.047 | 0.019 | - | - |
| battery | A2 | id | 0.0% | 0.018 | nan | nan | no | 3.4% | 0.018 | 0.025 | - | - |
| battery | A2 | shape_ood | 0.0% | 0.028 | nan | nan | no | 1.6% | 0.027 | 0.033 | - | - |
| battery | A2 | sequence_ood | 0.0% | 0.021 | nan | nan | no | 4.0% | 0.021 | 0.022 | - | - |

Envelopes:

- cartpole (seed 0): envelope lo [-485.492, -18.246, -0.722, -4.583] hi [499.29, 19.772, 0.698, 4.597]; train 1/99% [-360.741, -10.387, -0.148, -0.82] / [384.629, 10.905, 0.142, 0.822]; seq eval 1/99% [-83.991, -3.344, -0.104, -0.403] / [79.806, 2.996, 0.104, 0.405]; max |f(x + d e1) - f(x)| for d = 10, 100: {'10.0': 0.0, '100.0': 0.0} (|f| max 137)
- highway_traffic (seed 0): envelope lo [0.004, 0.004, 0.003, 0.004, 0.004] hi [0.106, 0.106, 0.13, 0.106, 0.107]; train 1/99% [0.024, 0.023, 0.028, 0.025, 0.024] / [0.084, 0.091, 0.104, 0.098, 0.095]; seq eval 1/99% [0.037, 0.036, 0.047, 0.045, 0.042] / [0.039, 0.049, 0.087, 0.088, 0.088]; max |f(x + d e1) - f(x)| for d = 10, 100: {'10.0': 32.44289456331575, '100.0': 3240.123969997325} (|f| max 0.00301)
- quad_tank (seed 0): envelope lo [5.09, 5.156, 0.124, 0.217] hi [18.839, 18.805, 6.369, 4.663]; train 1/99% [7.339, 7.186, 1.901, 1.82] / [15.675, 16.023, 5.86, 4.25]; seq eval 1/99% [10.653, 10.058, 4.838, 2.636] / [12.091, 11.931, 5.769, 4.103]; max |f(x + d e1) - f(x)| for d = 10, 100: {'10.0': 0.17408241670373106, '100.0': 0.8811300504591423} (|f| max 0.246)
- battery (seed 0): envelope lo [-0.001, -0.046, -0.046, -0.322, -0.343] hi [1.009, 0.045, 0.046, 4.29, 0.468]; train 1/99% [0.057, -0.036, -0.034, -0.01, -0.147] / [0.978, 0.035, 0.033, 3.2, 0.393]; seq eval 1/99% [0.267, -0.036, -0.018, 0.415, -0.045] / [0.813, 0.036, 0.019, 1.199, 0.333]; max |f(x + d e1) - f(x)| for d = 10, 100: {'10.0': 10.322454517013941, '100.0': 10.322454587076564} (|f| max 8.78)

|x1| (cart position for cartpole) median / 99% at t = 0, T/2, T (seed 0):

| plant | tier | t0 | T/2 | T |
|---|---|---|---|---|
| cartpole | id | 2.52 / 4.89 | 20.9 / 72.2 | 26.5 / 166 |
| cartpole | shape_ood | 2.5 / 4.92 | 23.9 / 69.1 | 44.7 / 140 |
| cartpole | sequence_ood | 2.5 / 4.87 | 15.5 / 67.4 | 26.6 / 92.8 |
| highway_traffic | id | 0.0553 / 0.0988 | 0.0376 / 0.0407 | 0.0371 / 0.0373 |
| highway_traffic | shape_ood | 0.0554 / 0.0982 | 0.0376 / 0.0406 | 0.0371 / 0.0372 |
| highway_traffic | sequence_ood | 0.0556 / 0.0988 | 0.0377 / 0.0406 | 0.0371 / 0.0373 |
| quad_tank | id | 12 / 17.8 | 11.1 / 12 | 11.6 / 12 |
| quad_tank | shape_ood | 11.9 / 17.8 | 11.1 / 11.9 | 11.6 / 11.9 |
| quad_tank | sequence_ood | 12 / 17.7 | 11.2 / 12.1 | 11.6 / 12.1 |
| battery | id | 0.551 / 0.793 | 0.571 / 0.851 | 0.572 / 0.942 |
| battery | shape_ood | 0.552 / 0.792 | 0.575 / 0.88 | 0.585 / 0.937 |
| battery | sequence_ood | 0.553 / 0.79 | 0.578 / 0.812 | 0.548 / 0.802 |