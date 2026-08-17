"""Server-rendered inline SVG charts for the dashboard.

Rendered on the server as plain SVG rather than handed to a charting library, for
one practical reason: this dashboard runs on a Raspberry Pi and gets opened from a
phone, sometimes on a connection that cannot reach a CDN. Inline SVG has no
runtime dependency, no bundle, and renders instantly.

Design follows the project's data-viz rules: one axis, recessive grid, thin marks,
2px lines, 4px rounded data-ends anchored to the baseline, a 2px surface gap
between adjacent bars, selective direct labels (never a number on every point),
and a hover layer. Colors come from the validated categorical palette — slot 1
blue for the primary series, slot 2 orange for the comparison series. Both modes
pass all six palette checks.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

# Categorical slots 1-2 (light / dark), from the validated palette.
SERIES_1 = "var(--series-1)"
SERIES_2 = "var(--series-2)"


def _fmt_krw(v: float) -> str:
    """Compact KRW for axis ticks: 억/만 read faster than 8 digits."""
    a = abs(v)
    if a >= 100_000_000:
        return f"{v / 100_000_000:.2f}억"
    if a >= 10_000:
        return f"{v / 10_000:,.0f}만"
    return f"{v:,.0f}"


def _nice_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    """Round tick values spanning [lo, hi]."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / count
    mag = 10 ** (len(str(int(abs(raw)))) - 1) if abs(raw) >= 1 else 1
    step = max(round(raw / mag) * mag, mag)
    start = (int(lo / step)) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            ticks.append(v)
        v += step
    return ticks or [lo, hi]


def equity_chart(
    rows: list[dict[str, Any]],
    *,
    width: int = 880,
    height: int = 260,
) -> str:
    """Equity curve. One series, so no legend — the panel title names it."""
    if len(rows) < 2:
        return (
            '<p class="empty">평가금액 기록이 아직 2개 미만입니다. '
            "봇을 실행하면 30분마다 기록됩니다.</p>"
        )

    pad_l, pad_r, pad_t, pad_b = 62, 58, 16, 26
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    values = [float(r["equity"]) for r in rows]
    lo_v, hi_v = min(values), max(values)
    if hi_v == lo_v:
        hi_v, lo_v = hi_v * 1.001 + 1, lo_v * 0.999 - 1
    span = hi_v - lo_v
    lo_v -= span * 0.08
    hi_v += span * 0.08

    n = len(rows)

    def x_of(i: int) -> float:
        return pad_l + (plot_w * i / (n - 1))

    def y_of(v: float) -> float:
        return pad_t + plot_h * (1 - (v - lo_v) / (hi_v - lo_v))

    pts = [(x_of(i), y_of(v)) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = (
        f"{pad_l},{pad_t + plot_h} "
        + line
        + f" {pad_l + plot_w},{pad_t + plot_h}"
    )

    # Recessive gridlines + tick labels on one axis only.
    grid = []
    for t in _nice_ticks(lo_v, hi_v):
        y = y_of(t)
        if not (pad_t - 1 <= y <= pad_t + plot_h + 1):
            continue
        grid.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'class="grid"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">'
            f"{_fmt_krw(t)}</text>"
        )

    # Date labels at the ends only — a label on every point is noise.
    def short_date(ts: str) -> str:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%m/%d")
        except ValueError:
            return ts[5:10]

    first_lbl = short_date(rows[0]["ts"])
    last_lbl = short_date(rows[-1]["ts"])

    # One direct label: the current value, at the end of the line.
    last_x, last_y = pts[-1]
    current = values[-1]
    change = (current / values[0] - 1) if values[0] else 0.0
    delta_class = "up" if change >= 0 else "down"

    hover = json.dumps(
        [
            {"x": round(x, 1), "y": round(y, 1), "v": v, "t": r["ts"][:16].replace("T", " ")}
            for (x, y), v, r in zip(pts, values, rows, strict=True)
        ]
    )

    return f"""
<figure class="chart">
  <svg viewBox="0 0 {width} {height}" class="viz" role="img"
       aria-label="평가금액 추이" data-points='{hover}'
       data-plot='{{"l":{pad_l},"r":{pad_l + plot_w},"t":{pad_t},"b":{pad_t + plot_h}}}'>
    <defs>
      <linearGradient id="eqfill" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%"  stop-color="var(--series-1)" stop-opacity="0.18"/>
        <stop offset="100%" stop-color="var(--series-1)" stop-opacity="0.01"/>
      </linearGradient>
    </defs>
    {"".join(grid)}
    <polygon points="{area}" fill="url(#eqfill)"/>
    <polyline points="{line}" fill="none" stroke="{SERIES_1}" stroke-width="2"
              stroke-linejoin="round" stroke-linecap="round"/>
    <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}"
          y2="{pad_t + plot_h}" class="axis"/>
    <text x="{pad_l}" y="{height - 8}" class="tick">{first_lbl}</text>
    <text x="{pad_l + plot_w}" y="{height - 8}" class="tick"
          text-anchor="end">{last_lbl}</text>
    <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="{SERIES_1}"
            stroke="var(--surface-1)" stroke-width="2"/>
    <text x="{last_x + 8:.1f}" y="{last_y + 4:.1f}" class="direct-label">
      {_fmt_krw(current)}
    </text>
    <g class="crosshair" style="display:none">
      <line class="cx" y1="{pad_t}" y2="{pad_t + plot_h}"/>
      <circle class="cdot" r="5" fill="{SERIES_1}" stroke="var(--surface-1)"
              stroke-width="2"/>
    </g>
  </svg>
  <div class="tooltip" hidden></div>
  <figcaption>
    기간 수익률 <span class="{delta_class}">{change:+.2%}</span>
    · {len(rows)}개 기록
  </figcaption>
</figure>
"""


def weights_chart(
    rows: list[dict[str, Any]], *, width: int = 880, bar_h: int = 22
) -> str:
    """Target vs actual weight per symbol.

    Two series, so a legend is present *and* every bar is directly labeled —
    identity never rests on color alone.
    """
    rows = [r for r in rows if (r.get("target") or 0) > 0.0005 or (r.get("actual") or 0) > 0.0005]
    if not rows:
        return '<p class="empty">목표 비중이 없습니다. (전액 현금 상태)</p>'

    rows = sorted(rows, key=lambda r: -max(r.get("target") or 0, r.get("actual") or 0))
    pad_l, pad_r, pad_t = 108, 92, 30
    row_h = bar_h * 2 + 8  # two bars + a 2px-scale gap, plus row spacing
    height = pad_t + len(rows) * row_h + 12
    plot_w = width - pad_l - pad_r

    hi = max(max(r.get("target") or 0, r.get("actual") or 0) for r in rows)
    hi = max(hi, 0.05) * 1.15

    parts: list[str] = []
    # Legend — always present for two series.
    parts.append(
        f'<g class="legend"><rect x="{pad_l}" y="6" width="10" height="10" rx="2" '
        f'fill="{SERIES_1}"/><text x="{pad_l + 16}" y="15" class="legend-t">목표</text>'
        f'<rect x="{pad_l + 62}" y="6" width="10" height="10" rx="2" fill="{SERIES_2}"/>'
        f'<text x="{pad_l + 78}" y="15" class="legend-t">실제</text></g>'
    )

    for i, r in enumerate(rows):
        y0 = pad_t + i * row_h
        tgt = float(r.get("target") or 0)
        act = float(r.get("actual") or 0)
        label = r.get("name") or r["symbol"]
        parts.append(
            f'<text x="{pad_l - 10}" y="{y0 + bar_h}" class="cat" '
            f'text-anchor="end">{_esc(label[:14])}</text>'
        )
        for j, (val, color) in enumerate(((tgt, SERIES_1), (act, SERIES_2))):
            w = max(plot_w * val / hi, 0.0)
            y = y0 + j * (bar_h + 2)  # 2px surface gap between adjacent bars
            # Rounded data-end only; the baseline end stays square.
            parts.append(
                f'<rect x="{pad_l}" y="{y}" width="{w:.1f}" height="{bar_h - 2}" '
                f'rx="4" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{pad_l + w + 8:.1f}" y="{y + bar_h - 7}" '
                f'class="bar-label">{val:.1%}</text>'
            )

    return f"""
<figure class="chart">
  <svg viewBox="0 0 {width} {height}" class="viz" role="img"
       aria-label="종목별 목표 대비 실제 비중">
    <line x1="{pad_l}" y1="{pad_t - 4}" x2="{pad_l}" y2="{height - 8}" class="axis"/>
    {"".join(parts)}
  </svg>
</figure>
"""


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
