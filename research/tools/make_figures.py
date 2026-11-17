"""Generate the SVG figures used by the README.

Only the Python standard library is used so the figures can be rebuilt on any
machine (no matplotlib / cairo dependency). Run from the repository root:

    python research/tools/make_figures.py
"""

from __future__ import annotations

import csv
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BLUE = "#1d4ed8"
RED = "#dc2626"
GRAY = "#9ca3af"


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=13, fill=INK, anchor="start", weight="400", extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" {extra}>'
        f"{esc(str(s))}</text>"
    )


def line(x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{d}/>'
    )


def rect(x, y, w, h, fill="#ffffff", stroke="#d1d5db", width=1.2, rx=8):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
    )


def arrow(x1, y1, x2, y2, stroke="#6b7280", width=1.8):
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}" marker-end="url(#arrow)"/>'
    )


def svg(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"<defs><marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" "
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/></marker></defs>\n'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n{body}\n</svg>\n'
    )


# --------------------------------------------------------------------------- #
# V6 training curves
# --------------------------------------------------------------------------- #

def training_curve():
    rows = []
    with (ROOT / "research/v6/iteration-history.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({k: float(v) for k, v in row.items()})
    rows.sort(key=lambda r: r["iteration"])

    W, H = 960, 380
    left, right, top, bottom = 66, 900, 64, 320
    xmin, xmax = 1, max(r["iteration"] for r in rows)
    y1min, y1max = -0.5, 0.3
    y2min, y2max = 0.0, 0.8

    def px(it):
        return left + (it - xmin) / (xmax - xmin) * (right - left)

    def py1(v):
        return bottom - (v - y1min) / (y1max - y1min) * (bottom - top)

    def py2(v):
        return bottom - (v - y2min) / (y2max - y2min) * (bottom - top)

    body = [text(24, 30, "V6 shared PPO: one objective, two behaviours", 17, INK, weight="600")]
    body.append(text(24, 48, "68 iterations, 4,200 s on one GPU, frozen opponent pool", 12, MUTED))

    for v in (-0.5, -0.3, -0.1, 0.1, 0.3):
        y = py1(v)
        body.append(line(left, y, right, y, GRID, 1))
        body.append(text(left - 8, y + 4, f"{v:+.1f}", 11, MUTED, "end"))
    for v in (0.0, 0.2, 0.4, 0.6, 0.8):
        y = py2(v)
        body.append(text(right + 8, y + 4, f"{v:.1f}", 11, MUTED))
    body.append(line(left, py1(0), right, py1(0), "#9ca3af", 1.2))
    body.append(line(left, bottom, right, bottom, "#9ca3af", 1.2))
    body.append(line(left, top, left, bottom, "#9ca3af", 1.2))
    body.append(line(right, top, right, bottom, "#9ca3af", 1.2))

    for it in range(0, int(xmax) + 1, 10):
        if it < xmin:
            continue
        x = px(it)
        body.append(line(x, bottom, x, bottom + 4, "#9ca3af", 1))
        body.append(text(x, bottom + 18, str(it), 10, MUTED, "middle"))
    body.append(text((left + right) / 2, H - 10, "PPO iteration", 12, MUTED, "middle"))

    for key, py, color, label in (
        ("net_kills", py1, BLUE, "netKills (left)"),
        ("suicides", py2, RED, "suicides (right)"),
    ):
        pts = " ".join(f"{px(r['iteration']):.1f},{py(r[key]):.1f}" for r in rows)
        body.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    best = max(rows, key=lambda r: r["fitness"])
    bx, by = px(best["iteration"]), py1(best["net_kills"])
    body.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4.5" fill="{BLUE}" stroke="#ffffff" stroke-width="1.5"/>')
    body.append(text(bx + 10, by - 10, f"iter {int(best['iteration'])}: netKills +{best['net_kills']:.3f}", 11, BLUE, weight="600"))
    body.append(text(bx + 10, by + 5, f"suicides {best['suicides']:.3f} - saved checkpoint", 10, MUTED))

    body.append(line(620, 26, 640, 26, BLUE, 2.4))
    body.append(text(646, 30, "netKills", 12, INK))
    body.append(line(720, 26, 740, 26, RED, 2.4))
    body.append(text(746, 30, "suicides", 12, INK))

    (ASSETS / "training-v6.svg").write_text(svg(W, H, "\n".join(body)), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Headline comparison
# --------------------------------------------------------------------------- #

def results_chart():
    W, H = 960, 410
    body = [text(24, 30, "The gap we closed, and the gap we did not", 17, INK, weight="600")]
    body.append(text(24, 48, "Frozen development split, 140 episodes, macro-averaged over n = 2..8", 12, MUTED))

    # --- win panel ---------------------------------------------------------
    wins = [
        ("Random", 1.19, GRAY),
        ("V5 win", 4.38, BLUE),
        ("V6 iter47", 7.74, BLUE),
        ("V7 distill", 16.41, BLUE),
        ("HY teacher", 30.19, INK),
    ]
    gate = 21.13
    px0, px1, py0, py1 = 70, 470, 330, 90
    ymax = 34.0

    def wy(v):
        return py0 - v / ymax * (py0 - py1)

    body.append(text(px0, 78, "Win rate (%)", 13, INK, weight="600"))
    for v in (0, 10, 20, 30):
        y = wy(v)
        body.append(line(px0, y, px1, y, GRID))
        body.append(text(px0 - 8, y + 4, str(v), 10, MUTED, "end"))
    body.append(line(px0, wy(gate), px1, wy(gate), RED, 1.6, "6 4"))
    body.append(text(px0 + 6, wy(gate) - 7, f"70% gate = {gate:.1f}%", 10, RED, "start", "600"))

    slot = (px1 - px0) / len(wins)
    for i, (name, value, color) in enumerate(wins):
        cx = px0 + slot * (i + 0.5)
        bar_w = slot * 0.52
        y = wy(value)
        body.append(rect(cx - bar_w / 2, y, bar_w, py0 - y, color, color, 0, 3))
        body.append(text(cx, y - 6, f"{value:.1f}", 10, INK, "middle", "600"))
        body.append(text(cx, py0 + 17, name, 10, MUTED, "middle"))

    # --- netKills panel ----------------------------------------------------
    nk = [
        ("Random", -0.873, GRAY),
        ("V6 iter47", 0.004, BLUE),
        ("V7 distill", 0.430, BLUE),
        ("HY teacher", 0.653, INK),
    ]
    gate_nk = 0.7 * 0.653
    qx0, qx1, qy0, qy1 = 550, 900, 330, 90
    ymin, ymax2 = -1.0, 0.8

    def ny(v):
        return qy0 - (v - ymin) / (ymax2 - ymin) * (qy0 - qy1)

    body.append(text(qx0, 78, "Net kills per round", 13, INK, weight="600"))
    for v in (-1.0, -0.5, 0.0, 0.5):
        y = ny(v)
        body.append(line(qx0, y, qx1, y, GRID))
        body.append(text(qx0 - 8, y + 4, f"{v:+.1f}", 10, MUTED, "end"))
    body.append(line(qx0, ny(0), qx1, ny(0), "#9ca3af", 1.2))
    body.append(line(qx0, ny(gate_nk), qx1, ny(gate_nk), RED, 1.6, "6 4"))
    body.append(text(qx0 + 6, ny(gate_nk) - 7, f"70% gate = {gate_nk:.2f}", 10, RED, "start", "600"))

    slot = (qx1 - qx0) / len(nk)
    for i, (name, value, color) in enumerate(nk):
        cx = qx0 + slot * (i + 0.5)
        bar_w = slot * 0.5
        y0, y1 = ny(0), ny(value)
        top, height = min(y0, y1), abs(y1 - y0)
        body.append(rect(cx - bar_w / 2, top, bar_w, height, color, color, 0, 3))
        label_y = y1 - 7 if value >= 0 else y1 + 15
        body.append(text(cx, label_y, f"{value:+.2f}", 10, INK, "middle", "600"))
        body.append(text(cx, py0 + 17, name, 10, MUTED, "middle"))

    body.append(text(24, 382, "Teacher = HY. Gate is 70% of teacher win rate; suicide ceiling is not shown. "
                    "V7 closes ~54% of the win-rate gap and ~66% of the net-kill gap.", 11, MUTED))
    (ASSETS / "results.svg").write_text(svg(W, H, "\n".join(body)), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def pipeline():
    W, H = 1000, 230
    stages = [
        ("G0", "contract audit", "engine + observation\n+ data receipt", "#f3f4f6", "#6b7280"),
        ("BC", "3k-step bootstrap", "behaviour prior only\n(not a gate)", "#eff6ff", BLUE),
        ("Shared PPO", "1 GPU, frag", "curriculum 2 -> 8\nopponent pool", "#eff6ff", BLUE),
        ("Fork", "2 GPUs", "win / frag\nfrom one checkpoint", "#eff6ff", BLUE),
        ("Dev gate", "140 episodes", "70% of teacher\n+ suicide ceiling", "#fef2f2", RED),
        ("Stop", "ship only if green", "never lowered\nthe thresholds", "#f3f4f6", "#6b7280"),
    ]
    body = [text(24, 30, "Fail-closed training pipeline", 17, INK, weight="600")]
    body.append(text(24, 48, "Every stage writes an auditable artifact; a failed gate stops the run instead of burning the next stage", 12, MUTED))

    x = 24
    box_w, box_h, y = 146, 92, 70
    gap = 17
    for i, (name, sub, note, fill, stroke) in enumerate(stages):
        body.append(rect(x, y, box_w, box_h, fill, stroke, 1.4, 10))
        body.append(text(x + box_w / 2, y + 24, name, 13, INK, "middle", "700"))
        body.append(text(x + box_w / 2, y + 42, sub, 10, MUTED, "middle"))
        for j, ln in enumerate(note.split("\n")):
            body.append(text(x + box_w / 2, y + 62 + j * 13, ln, 9.5, MUTED, "middle"))
        if i < len(stages) - 1:
            body.append(arrow(x + box_w + 2, y + box_h / 2, x + box_w + gap - 3, y + box_h / 2))
        x += box_w + gap

    body.append(text(24, 200, "The same frozen development split is used for model selection and for the final gate. "
                    "Sampled and greedy decoding must both pass; temperature is never tuned after training.", 11, MUTED))
    (ASSETS / "pipeline.svg").write_text(svg(W, H, "\n".join(body)), encoding="utf-8")


# --------------------------------------------------------------------------- #
# V6 architecture
# --------------------------------------------------------------------------- #

def architecture():
    W, H = 1100, 470
    body = [text(24, 30, "V6: relational tokens + conditional fire head", 17, INK, weight="600")]
    body.append(text(24, 48, "1,966-D observation -> 96 tokens -> 2-layer self-attention -> 51 motion logits + 1 fire gate", 12, MUTED))

    # observation
    body.append(rect(24, 84, 200, 64, "#f9fafb", "#6b7280", 1.4))
    body.append(text(124, 110, "Observation", 13, INK, "middle", "700"))
    body.append(text(124, 129, "1,966 floats / frame", 10, MUTED, "middle"))

    blocks = [
        ("self", "80"),
        ("map", "16 tokens"),
        ("enemies", "7"),
        ("bullets", "40"),
        ("aim cand.", "32"),
    ]
    by = 168
    for name, dim in blocks:
        body.append(rect(24, by, 200, 34, "#ffffff", "#d1d5db", 1.1, 6))
        body.append(text(36, by + 22, name, 11, INK))
        body.append(text(212, by + 22, dim, 10, MUTED, "end"))
        by += 40

    # block bus -> tokenizer
    for yy in (185, 225, 265, 305, 345):
        body.append(line(224, yy, 250, yy, "#9ca3af", 1.2))
    body.append(line(250, 116, 250, 345, "#9ca3af", 1.2))
    body.append(arrow(250, 116, 266, 116))

    body.append(rect(270, 84, 210, 64, "#eff6ff", BLUE, 1.4))
    body.append(text(375, 110, "Tokenize + type embedding", 12, INK, "middle", "700"))
    body.append(text(375, 129, "96 tokens (128-D)", 10, MUTED, "middle"))

    # attention
    body.append(arrow(484, 116, 516, 116))
    body.append(rect(520, 64, 200, 130, "#eef2ff", "#4f46e5", 1.4))
    body.append(text(620, 92, "2x Pre-LN", 12, INK, "middle", "700"))
    body.append(text(620, 110, "self-attention", 12, INK, "middle", "700"))
    body.append(text(620, 132, "128-D, 4 heads", 10, MUTED, "middle"))
    body.append(text(620, 150, "key-padding mask", 10, MUTED, "middle"))
    body.append(text(620, 168, "CLS + masked mean", 10, MUTED, "middle"))

    # trunk
    body.append(arrow(724, 130, 756, 130))
    body.append(rect(760, 90, 150, 80, "#f9fafb", "#6b7280", 1.4))
    body.append(text(835, 118, "Trunk", 13, INK, "middle", "700"))
    body.append(text(835, 137, "concat -> 768 -> 512", 10, MUTED, "middle"))
    body.append(text(835, 155, "2 residual blocks", 10, MUTED, "middle"))

    # heads
    heads = [
        ("motion head", "51 logits", 74),
        ("fire global", "1 logit", 144),
        ("fire cond.", "51 logits", 214),
    ]
    body.append(line(910, 130, 930, 130, "#9ca3af", 1.4))
    body.append(line(930, 94, 930, 234, "#9ca3af", 1.4))
    for name, dim, yy in heads:
        body.append(arrow(930, yy + 20, 946, yy + 20, "#9ca3af", 1.4))
        body.append(rect(950, yy, 126, 40, "#ffffff", "#d1d5db", 1.1, 6))
        body.append(text(960, yy + 25, name, 10.5, INK))
        body.append(text(1066, yy + 25, dim, 9.5, MUTED, "end"))

    body.append(rect(24, 386, 1050, 54, "#fffbeb", "#f59e0b", 1.2, 8))
    body.append(text(40, 408, "Conditional factorisation:", 11, INK, weight="700"))
    body.append(text(196, 408, "logit(m, f) = ( motion[m] + f * ( fire_global + fire_conditional[m] ) ) / T", 11, INK))
    body.append(text(40, 428, "V7 replaces the masked aim tokens with 64-direction traceShot labels: the solver becomes a label oracle, not an input feature.", 10, MUTED))

    (ASSETS / "architecture-v6.svg").write_text(svg(W, H, "\n".join(body)), encoding="utf-8")


if __name__ == "__main__":
    training_curve()
    results_chart()
    pipeline()
    architecture()
    for p in sorted(ASSETS.glob("*.svg")):
        print(p.name, p.stat().st_size)
