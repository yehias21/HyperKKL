# Case A: velocity and attitude estimation on a small quadrotor (fast, embedded, safety-critical)

Plant module: `hyperkkl/systems_casestudy.py::QuadPlanar` (registered `quad_planar`; config
`code/pkg/configs/systems/quad_planar.yaml`; case set-up `code/cs_common.py::CASES["A"]`).

## The practitioner's situation

A team flies a **0.5 kg inspection quadrotor** (5-inch class) along building facades and roof lines.
The flight controller is an **STM32H7 (ARM Cortex-M7 @ 480 MHz, single-precision FPU, 1 MB RAM, 2 MB
flash, no GPU)**. The rate/attitude loop runs at 400 Hz; the state estimator runs at **100 Hz** on
down-sampled IMU data (as PX4's EKF2 does). The position/velocity loop consumes the estimator's output.

**Per-step budget (stated before any measurement).** The estimator gets one 100 Hz slot of
**at most 150 us on one core** (1.5 % of the CPU, leaving the 400 Hz loop, the mixer, logging and
telemetry their share) and **at most 512 KB of flash for parameters (128 k float32 weights) and
32 KB of RAM** for its state. A method that does not fit is not deployable however accurate it is.

**What is measured** (outputs of `quad_planar`, physical noise std per sample):

| channel | sensor | rate | noise std |
|---|---|---|---|
| px | GNSS horizontal position | 10 Hz (sample-and-hold in the 100 Hz loop) | 0.25 m |
| pz | barometric altitude | 25 Hz (sample-and-hold) | 0.10 m |
| q | gyro pitch rate | 100 Hz | 0.03 rad/s (vibration-dominated) |
| a_m | body-x accelerometer (behind its 8 Hz DLPF) | 100 Hz | 0.15 m/s^2 (vibration) |

At the "realistic" knowledge level the noise is AR(1)-coloured (rho = 0.9) and heavy-tailed (Student-t,
nu = 3), as GNSS multipath and propeller vibration are. Gyro bias is assumed calibrated at arming;
its residual is part of the coloured noise.

**What must be estimated** (unmeasured): horizontal and vertical **velocity** (vx, vz) and **pitch**
(theta). The velocity feeds the position-hold loop (drift, overshoot near a wall); pitch feeds the
thrust compensation. Note that a quadrotor accelerometer does NOT measure tilt (it sees thrust plus
drag, not gravity); its body-x channel measures the **rotor drag**, -kd_x v_bx, i.e. velocity -- but
only if you model the drag (Leishman et al., IEEE CSM 2014).

**Model knowledge.** The team has the rigid-body + thrust model and its own firmware (controller gains,
thrust/torque limits, IMU filter settings are exact). Poorly known: **mass and inertia** (payload,
battery swaps), **rotor drag** (blade type, wear, wind) -- the declared UNMODELLED term of the
partial model is the rotor drag (the textbook model has none, so its accelerometer channel predicts
zero). Not modelled by anyone here: blade flapping, ground effect, gusts beyond the process noise
(0.02 x_std per sqrt(s) on every state: turbulence).

**Training data (the realistic budget).** A handful of motion-capture flights in the lab:
**60 episodes x 20 s = 20 min** of forced manoeuvres (training) + **12 x 20 s = 4 min** (validation,
model selection only) + **30 x 20 s = 10 min** of hover-hold from released initial conditions
(the zero-input set that HyperKKL's phase 1 and every transition-based fit use). **34 min of flight
in total**, with dense mocap ground truth for every state (velocity by differentiation of mocap
position is standard; here the simulator's true state is used).

**Inputs and operating envelope.** The input is the pilot / guidance set-point along an inclined
inspection path (horizontal set-point r, altitude set-point 0.3 r), |r| <= 3 m (the protocol's
bounded actuator box). The flight controller closes position, altitude and attitude loops (pitch
command saturated at 0.5 rad, thrust in [0, 2 m g], torque limit). Training manoeuvres = the
protocol's training families (steps, staircases, PRBS, squares, ramps, sines...); **OOD = new
manoeuvre styles**: chirps, filtered noise, triangles, sawtooths (shape_ood) and an aggressive
stop-and-go "drive" event sequence (sequence_ood).

**Robustness the team cares about** (id tier, realistic sensors of each level):
* a **poor initial guess** (estimator restarted mid-air from a far corner of the IC box),
* **sensor outliers** (1 % of samples off by 8 output-stds: GNSS jumps, vibration spikes),
* **GNSS loss** for 5 s (sample-and-hold of px: the filters skip stale samples, the learned
  observers see the frozen value), recovery afterwards,
* divergence (any |error| > 100 sigma or non-finite estimate).

## Knowledge spectrum (the team does not know where it is)

| level | modeller's model | true plant | sensor noise |
|---|---|---|---|
| exact | full physics, nominal parameters | = model | white Gaussian at the stated stds |
| mm10 | full physics | every physical parameter (m, Iyy, kd_x, kd_z) off by exp(+-0.1) | white |
| mm30 | full physics | off by exp(+-0.3) | white |
| partial | no rotor drag (UNMODELLED) | = nominal | white |
| realistic | no rotor drag | off by exp(+-0.3) | coloured Student-t (rho 0.9, nu 3) |

## The practitioner's requirement (fixed in PREREG.md before any observer run)
See PREREG.md section 2.
