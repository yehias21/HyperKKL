Reproduction (total NRMSE, stored clean vs re-evaluated `stored`, seeds 0-2):
- lora: 0.2869 / 0.2801 / 1.2246 vs 0.2869 / 0.2801 / 1.2246
- greybox_ukf: 0.0631 / 0.0641 / 0.0649 vs 0.0631 / 0.0641 / 0.0649
- A2: 1.2019 / 1.0751 / 1.1425 vs 1.2019 / 1.0751 / 1.1425
- A4: 0.5038 / 0.9246 / 0.2578 vs 0.5038 / 0.9246 / 0.2578

| model | tier | condition | total | unmeasured | x1 (meas.) | x2 cart vel | x3 (meas.) | x4 ang. vel |
|---|---|---|---|---|---|---|---|---|
| lora | id | stored | 0.3258 | 0.3164 | 0.1524 | 0.1715 | 0.2692 | 0.4613 |
| lora | id | ct_redraw | 0.3221 | 0.3129 | 0.1518 | 0.1719 | 0.2682 | 0.4540 |
| lora | id | noiseless | 0.2853 | 0.2732 | 0.1422 | 0.1649 | 0.2542 | 0.3815 |
| lora | id | gauss_white | 0.2923 | 0.2810 | 0.1447 | 0.1660 | 0.2555 | 0.3960 |
| lora | id | gauss_col | 0.3308 | 0.3225 | 0.1529 | 0.1735 | 0.2715 | 0.4715 |
| lora | shape_ood | stored | 0.3033 | 0.3059 | 0.1373 | 0.1289 | 0.2395 | 0.4829 |
| lora | shape_ood | ct_redraw | 0.2999 | 0.3024 | 0.1373 | 0.1295 | 0.2383 | 0.4753 |
| lora | shape_ood | noiseless | 0.2816 | 0.2815 | 0.1276 | 0.1239 | 0.2320 | 0.4390 |
| lora | shape_ood | gauss_white | 0.2836 | 0.2839 | 0.1304 | 0.1244 | 0.2320 | 0.4433 |
| lora | shape_ood | gauss_col | 0.3114 | 0.3136 | 0.1375 | 0.1292 | 0.2436 | 0.4979 |
| lora | sequence_ood | stored | 0.5972 | 0.6173 | 0.1443 | 0.1671 | 0.4565 | 1.0676 |
| lora | sequence_ood | ct_redraw | 0.5906 | 0.6109 | 0.1444 | 0.1667 | 0.4534 | 1.0551 |
| lora | sequence_ood | noiseless | 0.5794 | 0.5991 | 0.1357 | 0.1640 | 0.4453 | 1.0342 |
| lora | sequence_ood | gauss_white | 0.5812 | 0.6007 | 0.1383 | 0.1632 | 0.4454 | 1.0382 |
| lora | sequence_ood | gauss_col | 0.5998 | 0.6206 | 0.1446 | 0.1682 | 0.4581 | 1.0730 |
| greybox_ukf | id | stored | 0.0406 | 0.0478 | 0.0125 | 0.0192 | 0.0056 | 0.0763 |
| greybox_ukf | id | ct_redraw | 0.0409 | 0.0478 | 0.0129 | 0.0188 | 0.0057 | 0.0768 |
| greybox_ukf | id | noiseless | 0.0398 | 0.0470 | 0.0033 | 0.0178 | 0.0055 | 0.0763 |
| greybox_ukf | id | gauss_white | 0.0398 | 0.0470 | 0.0046 | 0.0177 | 0.0055 | 0.0763 |
| greybox_ukf | id | gauss_col | 0.0406 | 0.0474 | 0.0132 | 0.0184 | 0.0056 | 0.0764 |
| greybox_ukf | shape_ood | stored | 0.0390 | 0.0455 | 0.0122 | 0.0169 | 0.0058 | 0.0742 |
| greybox_ukf | shape_ood | ct_redraw | 0.0390 | 0.0456 | 0.0131 | 0.0172 | 0.0057 | 0.0739 |
| greybox_ukf | shape_ood | noiseless | 0.0380 | 0.0445 | 0.0030 | 0.0153 | 0.0056 | 0.0737 |
| greybox_ukf | shape_ood | gauss_white | 0.0381 | 0.0445 | 0.0043 | 0.0153 | 0.0056 | 0.0738 |
| greybox_ukf | shape_ood | gauss_col | 0.0389 | 0.0453 | 0.0126 | 0.0165 | 0.0057 | 0.0741 |
| greybox_ukf | sequence_ood | stored | 0.0640 | 0.0712 | 0.0131 | 0.0167 | 0.0079 | 0.1258 |
| greybox_ukf | sequence_ood | ct_redraw | 0.0638 | 0.0703 | 0.0129 | 0.0151 | 0.0079 | 0.1255 |
| greybox_ukf | sequence_ood | noiseless | 0.0633 | 0.0696 | 0.0030 | 0.0137 | 0.0077 | 0.1255 |
| greybox_ukf | sequence_ood | gauss_white | 0.0634 | 0.0695 | 0.0044 | 0.0135 | 0.0078 | 0.1255 |
| greybox_ukf | sequence_ood | gauss_col | 0.0639 | 0.0701 | 0.0130 | 0.0143 | 0.0078 | 0.1259 |
| A2 | id | stored | 0.5642 | 0.5085 | 0.3273 | 0.2765 | 0.5250 | 0.7405 |
| A2 | id | ct_redraw | 0.5651 | 0.5100 | 0.3273 | 0.2777 | 0.5246 | 0.7423 |
| A2 | id | noiseless | 0.5598 | 0.5039 | 0.3247 | 0.2729 | 0.5182 | 0.7349 |
| A2 | id | gauss_white | 0.5651 | 0.5110 | 0.3251 | 0.2739 | 0.5185 | 0.7482 |
| A2 | id | gauss_col | 0.5642 | 0.5099 | 0.3288 | 0.2766 | 0.5222 | 0.7432 |
| A2 | shape_ood | stored | 0.5574 | 0.5299 | 0.3139 | 0.1960 | 0.4849 | 0.8639 |
| A2 | shape_ood | ct_redraw | 0.5524 | 0.5248 | 0.3136 | 0.1950 | 0.4832 | 0.8546 |
| A2 | shape_ood | noiseless | 0.5544 | 0.5261 | 0.3136 | 0.1930 | 0.4800 | 0.8593 |
| A2 | shape_ood | gauss_white | 0.5554 | 0.5272 | 0.3134 | 0.1932 | 0.4822 | 0.8613 |
| A2 | shape_ood | gauss_col | 0.5572 | 0.5294 | 0.3139 | 0.1958 | 0.4871 | 0.8630 |
| A2 | sequence_ood | stored | 1.1398 | 1.1237 | 0.3030 | 0.2147 | 0.9374 | 2.0326 |
| A2 | sequence_ood | ct_redraw | 1.1458 | 1.1302 | 0.3017 | 0.2159 | 0.9406 | 2.0446 |
| A2 | sequence_ood | noiseless | 1.1310 | 1.1134 | 0.2984 | 0.2107 | 0.9327 | 2.0162 |
| A2 | sequence_ood | gauss_white | 1.1324 | 1.1155 | 0.2991 | 0.2120 | 0.9329 | 2.0190 |
| A2 | sequence_ood | gauss_col | 1.1445 | 1.1289 | 0.3031 | 0.2161 | 0.9402 | 2.0416 |
| A4 | id | stored | 0.2760 | 0.2726 | 0.1740 | 0.1584 | 0.1935 | 0.3868 |
| A4 | id | ct_redraw | 0.2673 | 0.2637 | 0.1730 | 0.1588 | 0.1914 | 0.3686 |
| A4 | id | noiseless | 0.2472 | 0.2445 | 0.1452 | 0.1441 | 0.1705 | 0.3450 |
| A4 | id | gauss_white | 0.2480 | 0.2462 | 0.1495 | 0.1460 | 0.1717 | 0.3464 |
| A4 | id | gauss_col | 0.2759 | 0.2727 | 0.1744 | 0.1587 | 0.1943 | 0.3866 |
| A4 | shape_ood | stored | 0.2415 | 0.2458 | 0.1584 | 0.1290 | 0.1693 | 0.3625 |
| A4 | shape_ood | ct_redraw | 0.2435 | 0.2481 | 0.1590 | 0.1290 | 0.1687 | 0.3672 |
| A4 | shape_ood | noiseless | 0.2273 | 0.2331 | 0.1292 | 0.1194 | 0.1598 | 0.3468 |
| A4 | shape_ood | gauss_white | 0.2276 | 0.2331 | 0.1350 | 0.1210 | 0.1595 | 0.3452 |
| A4 | shape_ood | gauss_col | 0.2440 | 0.2480 | 0.1606 | 0.1292 | 0.1689 | 0.3668 |
| A4 | sequence_ood | stored | 0.5620 | 0.5896 | 0.1673 | 0.1700 | 0.3807 | 1.0092 |
| A4 | sequence_ood | ct_redraw | 0.5653 | 0.5932 | 0.1664 | 0.1707 | 0.3822 | 1.0157 |
| A4 | sequence_ood | noiseless | 0.5453 | 0.5704 | 0.1383 | 0.1545 | 0.3601 | 0.9862 |
| A4 | sequence_ood | gauss_white | 0.5392 | 0.5648 | 0.1429 | 0.1566 | 0.3582 | 0.9730 |
| A4 | sequence_ood | gauss_col | 0.5651 | 0.5919 | 0.1669 | 0.1696 | 0.3836 | 1.0143 |

| model | l_dec_q95 (s0/s1/s2) | tier | unm drop noiseless vs stored | gauss white / ct redraw | gauss coloured / ct redraw | x2 (noiseless) / CRB(3/a) | x4 (noiseless) / CRB(3/a) |
|---|---|---|---|---|---|---|---|
| lora | 65 / 70 / 72 | id | +14% | 0.90 | 1.03 | 2.2 | 56 |
| lora | 65 / 70 / 72 | shape_ood | +8% | 0.94 | 1.04 | 1.6 | 65 |
| lora | 65 / 70 / 72 | sequence_ood | +3% | 0.98 | 1.02 | 2.2 | 152 |
| greybox_ukf | - | id | +2% | 0.98 | 0.99 | 0.2 | 11 |
| greybox_ukf | - | shape_ood | +2% | 0.98 | 0.99 | 0.2 | 11 |
| greybox_ukf | - | sequence_ood | +2% | 0.99 | 1.00 | 0.2 | 18 |
| A2 | 313 / 218 / 256 | id | +1% | 1.00 | 1.00 | 3.6 | 108 |
| A2 | 313 / 218 / 256 | shape_ood | +1% | 1.00 | 1.01 | 2.5 | 126 |
| A2 | 313 / 218 / 256 | sequence_ood | +1% | 0.99 | 1.00 | 2.8 | 296 |
| A4 | 251 / 281 / 290 | id | +10% | 0.93 | 1.03 | 1.9 | 51 |
| A4 | 251 / 281 / 290 | shape_ood | +5% | 0.94 | 1.00 | 1.6 | 51 |
| A4 | 251 / 281 / 290 | sequence_ood | +3% | 0.95 | 1.00 | 2.0 | 145 |

Bayesian PCRB per state (x_std units, eval_id, budgets.json): window 3/alpha = 2 s: 0.0059, 0.0762, 0.0020, 0.0068; full past: 0.0045, 0.0145, 0.0020, 0.0068.
