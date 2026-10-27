# Bibliography verification (2026-10-25)

Scope: the 13 `% TODO verify` entries of paper/refs.bib, the archive entries flagged VERIFY
(ramos2020, bernard2017, janny2021, buisson2023, niazi2023, peralez2021, peralez2022, peralez2024,
chauhan2024), and the journal/arXiv entries of theory/refs.bib. Keys are unchanged. Sources: Crossref
metadata (DOI records, `https://api.crossref.org/works/<doi>`), publisher and proceedings pages, arXiv.
After the edits, `bash paper/build.sh` reports 0 undefined references/citations and 0 BibTeX warnings.

## TODO-verify entries (all resolved, TODO markers removed)

| key | verified source | changes |
|---|---|---|
| julier2004unscented | https://doi.org/10.1109/JPROC.2003.823141 | correct; added DOI |
| gordon1993novel | https://doi.org/10.1049/ip-f-2.1993.0015 | correct (140(2):107--113); added DOI |
| kidger2020neuralcde | https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html | correct; added volume 33 |
| rubanova2019latentode | https://proceedings.neurips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html | title corrected to "Latent ordinary differential equations for irregularly-sampled time series"; added volume 32 |
| chen2018neuralode | https://proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html | correct; added volume 31 |
| revach2022kalmannet | https://doi.org/10.1109/TSP.2022.3158588 | correct; added DOI |
| haarnoja2016backpropkf | https://proceedings.neurips.cc/paper_files/paper/2016/file/697e382cfd25b07a3e62275d3ee132b3-Paper.pdf | correct; added volume 29 |
| kirchmeyer2022coda | https://proceedings.mlr.press/v162/kirchmeyer22a.html | correct (PMLR 162:11283--11301); added series/volume |
| williams2015edmd | https://doi.org/10.1007/s00332-015-9258-5 | correct; added DOI |
| lusch2018deep | https://doi.org/10.1038/s41467-018-07210-0 | correct (article 4950); added number 1, DOI |
| surana2016koopman | https://doi.org/10.1109/CDC.2016.7799268 | correct; added DOI |
| schuirmann1987tost | https://doi.org/10.1007/BF01068419 | correct; added DOI |
| huber1964robust | https://doi.org/10.1214/aoms/1177703732 | correct; added DOI |

## Archive entries flagged VERIFY

| key | verified source | changes |
|---|---|---|
| bernard2017 | https://doi.org/10.1137/15M1037755 | **volume 55 -> 56, year 2017 -> 2018**; added DOI (key kept) |
| ramos2020 | https://doi.org/10.1109/CDC42340.2020.9304163 | **first author corrected: "Luís André Ramos" -> "Louise da Costa Ramos"**; "Luiz" -> "Luis" Fernando Figueira da Silva; added DOI |
| janny2021 | https://doi.org/10.1109/CDC45484.2021.9683277 | correct; added DOI |
| buisson2023 | https://doi.org/10.1016/j.ifacol.2023.10.1730 | **author list corrected: Trimpe removed, Lukas Bahr added (Buisson-Fenet, Bahr, Morgenthaler, Di Meglio)**; now @article in IFAC-PapersOnLine 56(2):4061--4067; added DOI |
| niazi2023 | https://doi.org/10.23919/ACC55779.2023.10156294 | correct; added DOI |
| peralez2021 | https://doi.org/10.1109/CDC45484.2021.9683485 | correct; added DOI |
| peralez2022 | https://doi.org/10.1109/CDC51059.2022.9992516 | added pages 2105--2110 and DOI |
| peralez2024 | -- | **NOT FOUND** (see below); entry left as is with a `% NOT FOUND` comment |
| chauhan2024 | https://doi.org/10.1007/s10462-024-10862-8 | added number 9, article 250, DOI |

### peralez2024 -- NOT FOUND
"Learning-based observer design for nonlinear systems with inputs", Peralez & Nadri, IEEE Control
Systems Letters, 2024: no such paper in Crossref (all Peralez papers 2022+ listed), web search, or
citing papers. The only 2024 Peralez--Nadri observer paper is "Deep model-free KKL observer: A
switching approach", L4DC 2024, PMLR 242:929--940 (https://proceedings.mlr.press/v242/peralez24a.html),
which addresses autonomous systems (peaking/noise), not inputs. The sentence in
sections/related.tex ("\citet{peralez2024} learn an observer with inputs") is therefore unsupported;
the author must replace the reference or drop the claim.

## theory/refs.bib
Checked against Crossref: bernard2019luenberger, andrieu2006existence (10.1137/040617066),
brivadis2023further (10.1016/j.sysconle.2022.105429), boyd1985fading, leshno1993multilayer,
lohmiller1998contraction, reif1999stochastic, hochbruck2010exponential, korda2018linear,
proctor2016dmdc; arXiv: niazi2025kkl (2501.11655), shaaban2026hyperkkl (2603.29744). All correct; no
changes to theory/refs.bib. Not separately checked (conference/book records, not flagged): zeng2024expressive,
miller2019stable, virmaux2018lipschitz, fazlyab2019efficient, khalil2002nonlinear, andersonmoore1979.

## Note (not flagged, not changed)
woelk2026 (arXiv:2602.24252, verified title/authors) appears to now have a journal version on
ScienceDirect (pii S0959152426001149, Journal of Process Control); consider updating at camera-ready.

### peralez2024 — resolved by the lead (2026-10-26)
Entry replaced by the verified paper: Peralez & Nadri, "Deep model-free KKL observer: A switching approach", L4DC 2024, PMLR 242:929–940 (https://proceedings.mlr.press/v242/peralez24a.html). The sentence in sections/related.tex now describes it correctly (model-free KKL learning for autonomous systems) instead of the unsupported "observer with inputs" claim.
