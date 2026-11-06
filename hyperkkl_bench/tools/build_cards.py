"""Write docs/cards/<plant>.md and docs/cards/INDEX.md from the metadata + difficulty resource."""
from pathlib import Path
import hyperkkl_bench as hkb
from hyperkkl_bench.cards import card_markdown, plant_card

OUT = Path(__file__).resolve().parents[1] / "docs" / "cards"
OUT.mkdir(parents=True, exist_ok=True)
rows = ["# Plant cards", "", "| plant | role | n_x | n_y | group | window/full info ratio (white) | repairability | Gramian cond | bounded |",
        "|---|---|---|---|---|---|---|---|---|"]
for p in hkb.list_plants(include_legacy=True) + hkb.list_plants(role="deployment") + [q for q in hkb.registry.PLANTS if hkb.registry.PLANTS[q]["role"] == "scaling"]:
    (OUT / f"{p}.md").write_text(card_markdown(p))
    c = plant_card(p); d = c["difficulty"] or {}
    ib = (d.get("info_budget") or {}).get("white") or {}
    rp = d.get("repairability") or {}
    ob = d.get("observability") or {}
    f = lambda v, fm: "n/a" if v is None else fm.format(v)
    rows.append(f"| [{p}]({p}.md) | {c['status']['role']} | {c['n_x']} | {c['n_y']} | {c.get('group')} | "
                f"{f(ib.get('window_ratio'), '{:.2f}')} | {f(rp.get('rho'), '{:.2f}')} | {f(ob.get('gramian_cond_median'), '{:.2g}')} | "
                f"{(d.get('inclusion') or {}).get('bounded')} |")
(OUT / "INDEX.md").write_text("\n".join(rows) + "\n")
print("wrote", OUT)
