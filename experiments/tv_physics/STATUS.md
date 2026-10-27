2026-10-26 22:02 Dubai — started; folder created; beginning correctness audit
2026-10-26 22:21 Dubai — code (tv.py, run_tv.py, tests) written; OFF seed0 reproduction + residual-scale probe launched on GPUs 1-4
2026-10-26 22:25 Dubai — PREREG.md written (22:25); OFF seed0 reproduced campaign A4+D3 10/10; launched jobs1 (30 TVr seed-0 grid, 30 P1OFF, 30 TOr1) via dispatch.py on GPUs 0-4
2026-10-26 22:54 Dubai — OFF seeds1-2 evaluated (cached campaign models); 15/30 seed-0 TVr runs done; unit tests 5/7 passed so far (fixture tests running on CPU)
2026-10-26 23:01 Dubai — seed-0 grid done; lambda~ selected on validation (selection.json); launched jobs2 (20 TVr chosen, seeds 1-2) as a second dispatcher (combined load kept <=3/GPU? see note)
2026-10-26 23:03 Dubai — deviation (conduct only): the first dispatcher mis-flagged my own relative-path .venv processes as foreign and moved to GPUs 5-7 (3/GPU there); a second dispatcher briefly overloaded GPUs 0-4 (orphans killed within 30 s, no results written). Replaced by one dispatcher with global per-GPU counting (code/dispatch.py, jobs_all.txt). No change to the protocol.
2026-10-26 23:24 Dubai — P1OFF 30/30 done; TVr seeds1-2 and TOr1 running
2026-10-27 00:00 Dubai — all 110 runs done (OFF 30 reproduce campaign 30/30; TVr grid 30 + 20; P1OFF 30; TOr1 30); analysis.json + tables.md written
2026-10-27 00:21 Dubai — unit tests 7/7 pass (GPU fixture); AUDIT.md, RESULTS.md final. Verdicts: TV loss does not help (0/5,0/5; Holm 0/10); phase-1 PDE does not help (0/5 better; lorenz/partial PDE worse). DONE.
