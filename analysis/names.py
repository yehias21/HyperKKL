"""One place for how methods are NAMED and DRAWN in every figure and table of the paper
(visual fixer, round 1; used by analysis/figures.py, analysis/tables.py, hyperkkl/realdata_report.py).

Raw identifiers (run keys such as 'lora', 'kkl_mb', 'greybox_ekf_oracle') stay in the data; every legend,
tick label and table cell goes through `disp()`. The headline naming of the learned variants
(HyperKKL = the 'lora' runs, the dense 'hyper' ablation) is owned by analysis/store.py
(store.DISPLAY / store.SHORT_DISPLAY) and wins over the table below."""

from __future__ import annotations

DISPLAY = {
    # learned KKL observers (conditioning locations)
    "autonomous": "autonomous", "latent": "latent", "concat": "concat", "film": "FiLM", "lora": "LoRA",
    "hyper": "hyper", "ws_best": "WS-best",
    # data-driven same-data baselines
    "nlox": "NLOX", "koopman_kf": "Koopman KF", "koopman_kf_bilinear": "Koopman KF (bilinear)",
    "greybox_ekf": "grey-box EKF", "greybox_ukf": "grey-box UKF", "greybox_ekf_oracle": "grey-box EKF-oracle",
    # model-based filters
    "ekf": "EKF", "ukf": "UKF", "pf": "PF", "kkl_mb": "KKL-MB", "ekf_huber": "EKF-Huber", "ukf_huber": "UKF-Huber",
    "ekf_oracle": "EKF-oracle", "ukf_oracle": "UKF-oracle", "ekf_oracle_huber": "EKF-oracle-Huber",
    "cc": "coulomb counting",
}

# compact forms for crowded cells and legends (explained once in a caption)
SHORT = {
    "ekf_oracle": "EKF-or", "ukf_oracle": "UKF-or", "ekf_huber": "EKF-Hu", "ukf_huber": "UKF-Hu",
    "ekf_oracle_huber": "EKF-orHu", "koopman_kf": "Koopman KF", "koopman_kf_bilinear": "Koopman-bi",
    "greybox_ekf": "GB-EKF", "greybox_ukf": "GB-UKF", "greybox_ekf_oracle": "GB-EKF-or",
}


def disp(m, short: bool = False) -> str:
    try:
        from . import store as _S
        own = _S.SHORT_DISPLAY if short else _S.DISPLAY
        if m in own:
            return own[m]
    except (ImportError, AttributeError):
        pass
    if short:
        return SHORT.get(m, DISPLAY.get(m, str(m)))
    return DISPLAY.get(m, str(m))


def short(m) -> str:
    return disp(m, short=True)


# Okabe-Ito; one colour per method, everywhere
COLOR = {"autonomous": "#999999", "latent": "#E69F00", "concat": "#56B4E9", "film": "#009E73",
         "lora": "#0072B2", "hyper": "#D55E00",
         # data-driven same-data baselines: NLOX pink, Koopman purple, grey-box olive
         "nlox": "#CC79A7", "koopman_kf": "#AA4499", "koopman_kf_bilinear": "#AA4499",
         "greybox_ekf": "#999933", "greybox_ukf": "#999933", "greybox_ekf_oracle": "#999933",
         # model-based filters: black (marker shape tells them apart)
         "ekf": "#000000", "ukf": "#000000", "pf": "#000000", "kkl_mb": "#000000",
         "ekf_huber": "#000000", "ukf_huber": "#000000", "ekf_oracle": "#000000", "ukf_oracle": "#000000",
         "ekf_oracle_huber": "#000000"}

# redundant shape encoding of the learned methods (greyscale / small markers)
METHOD_MARKER = {"autonomous": "o", "latent": "s", "concat": "D", "film": "v", "lora": "o", "hyper": "^",
                 "ws_best": "o", "nlox": "D", "koopman_kf": "D", "koopman_kf_bilinear": "D",
                 "greybox_ekf": "D", "greybox_ukf": "D", "greybox_ekf_oracle": "D"}

# model-based filters: one marker shape each
FILTER_MARKER = {"ekf": "o", "ukf": "s", "pf": "^", "kkl_mb": "v", "ekf_huber": "D", "ukf_huber": "d",
                 "ekf_oracle": "P", "ukf_oracle": "X", "ekf_oracle_huber": "*"}
