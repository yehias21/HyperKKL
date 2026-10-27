# VISUALS_LOG — visual-quality fixes (no numbers, claims or verdicts changed)

## Round 1 (2026-10-26, visual fixer), part A: done while the HyperKKL reframe held figures.py/tables.py/store.py/sections
| issue id | what changed | file |
|---|---|---|
| fig10-real-palette-and-raster | Real-data figures redrawn in the paper palette (latent #E69F00, concat #56B4E9, HyperKKL/lora #0072B2, dense/hyper #D55E00, WS-best = #0072B2 with a white hatch, EKF solid black, UKF white with black cross-hatch / grey dashed line, CC grey dotted, reference light-grey wide line); vector PDF at the printed size (each panel 3.2x2.6 in, 7 pt text, 6.5 pt ticks/legend); legends moved above the axes (no bar overlap); ASCII titles dropped (their content is now in the LaTeX caption). New `replot()` (`python -m hyperkkl.realdata_report`) redraws from the cache and the existing results.json without rewriting results.json; `report()` uses the same `figures()`. | hyperkkl/realdata_report.py |
| fig10-real-palette-and-raster | Includes switched to us06_window.pdf / rmse_test25.pdf at 0.49\textwidth (scale ~1); caption gains the settled-window / prior / seeds information formerly in the panel titles. | paper/appendix/realdata.tex |
| float-order-table9 | \FloatBarrier after the id summary table, so Table 9 prints before longtables 10/11. | paper/appendix/tables.tex |
| orphan-subsection-headings | \clearpage before F.2 (C1 tables); C2 figures [p] -> [tbp] so F.4's heading shares a page with its first figure. | paper/appendix/tables.tex |
| floatpage-centering | \@fptop=0pt, \@fpsep=12pt plus 2fil, \@fpbot=0pt plus 1fil: float pages are top-aligned. | paper/main.tex |
