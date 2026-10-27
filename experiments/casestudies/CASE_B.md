# Case B: concentration estimation in an exothermic CSTR (slow process, scarce labels)

Plant module: `hyperkkl/systems_casestudy.py::ExothermicCSTR` (registered `cstr_exo`; config
`code/pkg/configs/systems/cstr_exo.yaml`; case set-up `code/cs_common.py::CASES["B"]`).

Why a CSTR and not a fed-batch fermenter: the project already has a (continuous, Monod) bioreactor
plant; an exothermic CSTR is a different physics class (Arrhenius exponential nonlinearity, energy
balance, a jacket), and it is the textbook case where concentration estimation matters for **safety**
(thermal runaway), not only for quality.

## The practitioner's situation

A specialty-chemicals plant runs a **jacketed continuous reactor** (A -> B, exothermic, residence
time 30 min, adiabatic temperature rise 100 K). It is controlled by a **PLC / industrial PC**
(e.g. an x86 Atom/Celeron box next to an S7-1500), **CPU only**, with a **10 s sample time**.
The operators move the **coolant flow** (input u = relative change of the nominal coolant flow,
|u| <= 0.5) to follow production campaigns; the feed is fixed (measured flow, composition and
temperature).

**Per-step budget (generous).** One estimator update must finish in **<= 100 ms on one CPU core**
(1 % of the 10 s cycle) and fit in **<= 50 MB**. Every method considered here fits; the deciding
questions are accuracy with scarce labels, robustness, and how much the plant engineers must trust a
black box.

**What is measured:** reactor temperature T and jacket temperature T_j (RTDs, 0.1 K noise per sample;
AR(1)-coloured Student-t at the realistic level: mixing and flow fluctuations). **What must be
estimated:** the reactant concentration **C_A**, which sets the heat-generation potential: if coolant
is lost while C_A is high, the reactor runs away. There is no online analyser.

**Model knowledge.** Mass and energy balances are textbook; the Arrhenius parameters are not:
**pre-exponential factors / rate constants and heats of reaction +-10-30 %, activation energies
+-2.5-7.5 %** (the 30 % level perturbs E/R by exp(+-0.075): a 30 % error on E would move k(T) by
three orders of magnitude, not a credible prior); heat-transfer coefficients (fouling) +-10-30 %.
The declared **UNMODELLED** term: a **side reaction A -> D** with a higher activation energy
(E2/R = 16000 K vs 8000 K) and a larger heat of reaction -- negligible near the normal operating
point (10 % of the main rate at 335 K), ~30 % in the hot, low-coolant corner, exactly where a wrong
model is most dangerous (the partial model puts the minimum-coolant steady state at 344 K /
0.63 mol/L instead of 357 K / 0.23 mol/L).

**Training data (the realistic budget).** Three historical production campaigns:
**27 shifts x 8 h = 9 days** of operating data (training), **6 shifts = 2 days** (validation, model
selection only), **6 shifts** at constant nominal coolant flow from start-up states (the zero-input
set). Temperatures are logged at 10 s. **Concentration ground truth only from lab assays: one sample
every 45 +- 15 min** (uniform jitter), assay error 0.05 x_std (~0.01 mol/L) -- **about 10 assays per
shift, ~270 for training**. Masked losses (unlabelled samples contribute nothing) are used wherever a
method can: HyperKKL (phase-1 labels at assay times, phase-2 decoder loss), GRU, NLOX (masked TBPTT),
UDE / NODE (rollouts starting at an assay, measured temperatures dense, C_A only at the next assays,
K = 60 min windows). **The one-step grey-box residual cannot use sparse labels** (it needs the full
state at consecutive samples): it gets the practitioner's standard workaround, **linear interpolation
of the assays** as a dense pseudo-label. Validation (selection) also uses assay-time labels only.
Evaluation uses the simulator's dense truth.

**Inputs and campaigns.** Training = the protocol's training families at t_ref = 20 min (coolant
steps, staircases, PRBS-like campaign changes, ramps, sines over hours); **OOD = new operating
campaigns**: chirps / filtered-noise / triangle / sawtooth coolant programmes (shape_ood) and an
aggressive stop-and-go event sequence (sequence_ood). Episodes: one 8-h shift (480 min, N = 2880
samples); 64 per tier.

**Robustness the plant engineers care about** (id tier):
* **poor initial guess** (estimator restarted after a PLC reboot, prior at a far corner),
* **sensor outliers** (1 % of samples off by 8 output-stds: electrical spikes),
* **sensor loss** (reactor-temperature reading frozen for 30 min: transmitter fault / comms loss;
  the filters skip stale samples, the learned observers see the frozen value), recovery afterwards,
* divergence.

## Knowledge spectrum

| level | modeller's model | true plant | sensor noise |
|---|---|---|---|
| exact | both reactions, nominal parameters | = model | white Gaussian 0.1 K |
| mm10 | both reactions | rate constants, heats, UA off by exp(+-0.1), E/R by exp(+-0.025) | white |
| mm30 | both reactions | off by exp(+-0.3), E/R by exp(+-0.075) | white |
| partial | main reaction only (UNMODELLED side reaction) | nominal | white |
| realistic | main reaction only | off by exp(+-0.3) / exp(+-0.075) | coloured Student-t (rho 0.9, nu 3) |

## The practitioner's requirement (fixed in PREREG.md before any observer run)
See PREREG.md section 2.
