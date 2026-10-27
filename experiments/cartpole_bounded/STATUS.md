2026-10-26 18:20 Dubai  start; Part A budgets_c3 launched
2026-10-26 18:23 Dubai  INCLUSION.md rule written (before computing)
2026-10-26 18:27 Dubai  Part A done: BUDGET_FIX.md (cartpole id flips to CANNOT; no violations)
2026-10-26 18:28 Dubai  inclusion.py launched on the 12 existing plants (CPU)
2026-10-26 18:46 Dubai  systems_bounded.py (cartpole_b K=(-0.8,-0.6,..); two_link_arm_b elbow spring k_e=m2 g lc2), YAMLs, tests 7/7 pass; nodes TEST-C1-B/TEST-C3-B created (not queued); budget_new + inclusion for new plants launched
2026-10-26 18:51 Dubai  DESIGN.md written (designs fixed; no observer run yet)
2026-10-26 18:51 Dubai  PREREG.md written (before any observer run)
2026-10-26 18:52 Dubai  smoke tests launched (GPUs 0,1,2)
2026-10-26 19:06 Dubai  smoke OK: TEST-C1-B cartpole_b, TEST-C3-B cartpole_b (all 16 methods). two_link_arm_b: IC box re-centred on hanging pose after inclusion v1 showed shoulder winding (|q1| 16.6) -- checks rerunning, arm runs held until DESIGN/PREREG updated
2026-10-26 16:06 Dubai  launcher started: 9 jobs
2026-10-26 16:06 Dubai  start C3_cartpole_b_0 on gpu 0 (try 1)
2026-10-26 16:06 Dubai  start C3_cartpole_b_1 on gpu 1 (try 1)
2026-10-26 19:06 Dubai  NOTE: lines from the cartpole_b launcher (pid started 19:06 Dubai) carry SERVER time (CET, Dubai-3h) despite the 'Dubai' label; fixed for later launchers
2026-10-26 16:06 Dubai  start C3_cartpole_b_2 on gpu 2 (try 1)
2026-10-26 16:07 Dubai  start C1_cartpole_b_0 on gpu 3 (try 1)
2026-10-26 16:07 Dubai  start C1_cartpole_b_1 on gpu 4 (try 1)
2026-10-26 19:07 Dubai  two_link_arm_b rev2 (IC box on hanging pose) checked; DESIGN/PREREG addendum/INCLUSION verdict written. Original two_link_arm passes inclusion -> stays; arm_b optional, lowest priority
2026-10-26 19:07 Dubai  launcher started: 9 jobs
2026-10-26 19:07 Dubai  start C3_two_link_arm_b_0 on gpu 5 (try 1)
2026-10-26 16:07 Dubai  start C1_cartpole_b_2 on gpu 5 (try 1)
2026-10-26 19:07 Dubai  start C3_two_link_arm_b_1 on gpu 6 (try 1)
2026-10-26 19:08 Dubai  start C3_two_link_arm_b_2 on gpu 7 (try 1)
2026-10-26 19:08 Dubai  start C1_two_link_arm_b_0 on gpu 0 (try 1)
2026-10-26 19:08 Dubai  start C1_two_link_arm_b_1 on gpu 1 (try 1)
2026-10-26 19:09 Dubai  start C1_two_link_arm_b_2 on gpu 2 (try 1)
2026-10-26 16:17 Dubai  done C1_cartpole_b_0 (gpu 3)
2026-10-26 16:26 Dubai  done C1_cartpole_b_1 (gpu 4)
2026-10-26 16:29 Dubai  cuFFT transient on C1_cartpole_b_2, retry
2026-10-26 16:29 Dubai  start C1_cartpole_b_2 on gpu 3 (try 2)
2026-10-26 16:57 Dubai  done C1_cartpole_b_2 (gpu 3)
2026-10-26 17:03 Dubai  done C3_cartpole_b_1 (gpu 1)
2026-10-26 17:03 Dubai  start SOAP_cartpole_b_1 on gpu 1 (try 1)
2026-10-26 17:06 Dubai  done C3_cartpole_b_2 (gpu 2)
2026-10-26 17:06 Dubai  start SOAP_cartpole_b_2 on gpu 2 (try 1)
2026-10-26 17:07 Dubai  done C3_cartpole_b_0 (gpu 0)
2026-10-26 17:07 Dubai  start SOAP_cartpole_b_0 on gpu 0 (try 1)
2026-10-26 20:14 Dubai  done C1_two_link_arm_b_1 (gpu 1)
2026-10-26 17:14 Dubai  done SOAP_cartpole_b_1 (gpu 1)
2026-10-26 20:23 Dubai  done C1_two_link_arm_b_2 (gpu 2)
2026-10-26 17:25 Dubai  done SOAP_cartpole_b_2 (gpu 2)
2026-10-26 20:28 Dubai  done C1_two_link_arm_b_0 (gpu 0)
2026-10-26 17:29 Dubai  done SOAP_cartpole_b_0 (gpu 0)
2026-10-26 17:29 Dubai  launcher finished
2026-10-26 20:30 Dubai  cartpole_b done (TEST-C1-B, TEST-C3-B, SOAP A2/A4, seeds 0-2): A4 wins vs grey-box id 0.33x, seq 0.71x; RESULTS.md written. two_link_arm_b (optional) C3/SOAP still running under launcher
2026-10-26 21:12 Dubai  cuFFT transient on C3_two_link_arm_b_1, retry
2026-10-26 21:12 Dubai  start C3_two_link_arm_b_1 on gpu 0 (try 2)
2026-10-26 23:49 Dubai  done C3_two_link_arm_b_1 (gpu 0)
2026-10-26 23:49 Dubai  start SOAP_two_link_arm_b_1 on gpu 0 (try 1)
2026-10-27 00:13 Dubai  done SOAP_two_link_arm_b_1 (gpu 0)
2026-10-27 00:14 Dubai  done C3_two_link_arm_b_2 (gpu 7)
2026-10-27 00:14 Dubai  start SOAP_two_link_arm_b_2 on gpu 0 (try 1)
2026-10-27 00:17 Dubai  done C3_two_link_arm_b_0 (gpu 5)
2026-10-27 00:17 Dubai  start SOAP_two_link_arm_b_0 on gpu 1 (try 1)
2026-10-27 00:27 Dubai  done SOAP_two_link_arm_b_2 (gpu 0)
2026-10-27 00:31 Dubai  done SOAP_two_link_arm_b_0 (gpu 1)
2026-10-27 00:31 Dubai  launcher finished
