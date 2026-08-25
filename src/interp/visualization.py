# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Publication-quality visualization for interpretability results
#
# Oral papers have KILLER figures. This module generates:
# 1. Radar charts: JEPA vs baseline across metric categories
# 2. Heatmaps: layer-wise metric evolution
# 3. Information plane plots
# 4. Probing complexity curves
# 5. Training stability plots
# 6. Statistical comparison bar charts with error bars
#
# All plots use inline SVG (no external dependencies) so they
# render in any environment, including Kaggle notebooks.

import math
from pathlib import Path


def _svg_header(width=800, height=600):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'


def _svg_footer():
    return "</svg>\n"


def _svg_text(x, y, text, size=12, anchor="middle", color="black", weight="normal"):
    from html import escape

    safe = escape(str(text), quote=False)  # & < > must not break the SVG tree
    return f'  <text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{safe}</text>\n'


def _svg_line(x1, y1, x2, y2, color="black", width=1):
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>\n'


def _svg_rect(x, y, w, h, fill="steelblue", opacity=0.8):
    return f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" opacity="{opacity}"/>\n'


def _svg_circle(cx, cy, r, fill="red", opacity=0.8):
    return f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" opacity="{opacity}"/>\n'


def _svg_polygon(points, fill="steelblue", opacity=0.3, stroke="steelblue", stroke_width=1):
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'  <polygon points="{pts}" fill="{fill}" opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}"/>\n'


def radar_chart(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    title="JEPA vs Baseline",
    labels=None,
    width=600,
    height=600,
    output_path=None,
):
    """Radar/spider chart comparing JEPA vs baseline across metrics.

    Each metric is normalized to [0, 1] where 1 = better.
    JEPA polygon in blue, baseline in orange.

    Args:
        metrics: {metric_name: value} for JEPA
        baseline_metrics: {metric_name: value} for baseline
        title: chart title
        labels: optional {metric_name: display_label}
        output_path: if provided, save SVG to this path
    """
    names = [n for n in metrics if n in baseline_metrics]
    n = len(names)
    if n < 3:
        return None

    # Normalize to [0, 1] — higher = better
    # For metrics where lower is better, invert
    lower_is_better = {
        "anisotropy",
        "collapsed_dim_ratio",
        "mean_pairwise_cosine",
        "coherence",
        "condition_number",
        "total_compression",
        "mean_psi",
        "interference_ratio",
        "cv",
    }

    all_vals = []
    for name in names:
        all_vals.extend([metrics[name], baseline_metrics[name]])
    max_val = max(abs(v) for v in all_vals) if all_vals else 1
    max_val = max(max_val, 1e-10)

    def norm(name, val):
        v = abs(val) / max_val
        if name in lower_is_better or any(k in name for k in lower_is_better):
            v = 1.0 - v
        return max(0, min(1, v))

    cx, cy = width // 2, height // 2 + 20
    max_r = min(width, height) // 2 - 60

    svg = _svg_header(width, height)
    svg += _svg_text(cx, 25, title, size=16, weight="bold")

    # Draw axes and labels
    for i, name in enumerate(names):
        angle = 2 * math.pi * i / n - math.pi / 2
        ex = cx + max_r * math.cos(angle)
        ey = cy + max_r * math.sin(angle)
        svg += _svg_line(cx, cy, ex, ey, color="#cccccc", width=1)

        label = labels.get(name, name) if labels else name
        lx = cx + (max_r + 30) * math.cos(angle)
        ly = cy + (max_r + 30) * math.sin(angle)
        svg += _svg_text(lx, ly, label, size=9)

    # Draw grid circles
    for r_frac in [0.25, 0.5, 0.75, 1.0]:
        r = max_r * r_frac
        points = []
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        pts_str = " ".join(f"{x},{y}" for x, y in points)
        svg += f'  <polygon points="{pts_str}" fill="none" stroke="#eeeeee" stroke-width="1"/>\n'

    # Draw JEPA polygon
    jepa_points = []
    for i, name in enumerate(names):
        angle = 2 * math.pi * i / n - math.pi / 2
        r = max_r * norm(name, metrics[name])
        jepa_points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    svg += _svg_polygon(jepa_points, fill="#4285f4", opacity=0.25, stroke="#4285f4", stroke_width=2)

    # Draw baseline polygon
    base_points = []
    for i, name in enumerate(names):
        angle = 2 * math.pi * i / n - math.pi / 2
        r = max_r * norm(name, baseline_metrics[name])
        base_points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    svg += _svg_polygon(base_points, fill="#f4a142", opacity=0.2, stroke="#f4a142", stroke_width=2)

    # Legend
    svg += _svg_rect(30, height - 50, 15, 15, fill="#4285f4")
    svg += _svg_text(55, height - 38, "JEPA", size=11, anchor="start")
    svg += _svg_rect(130, height - 50, 15, 15, fill="#f4a142")
    svg += _svg_text(155, height - 38, "Baseline", size=11, anchor="start")

    svg += _svg_footer()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(svg)
    return svg


def layer_heatmap(
    layer_data: dict[str, list[float]],
    title="Layer-wise Metrics",
    width=800,
    height=400,
    output_path=None,
):
    """Heatmap of metric values across layers.

    Args:
        layer_data: {metric_name: [values per layer]}
        output_path: save SVG here
    """
    metrics = list(layer_data.keys())
    n_metrics = len(metrics)
    if n_metrics == 0:
        return None

    n_layers = max(len(v) for v in layer_data.values())
    cell_w = min(50, (width - 200) // n_layers)
    cell_h = min(30, (height - 80) // n_metrics)
    cell_w * n_layers
    total_h = cell_h * n_metrics

    # Find global min/max for color mapping
    all_vals = [v for vals in layer_data.values() for v in vals]
    vmin = min(all_vals) if all_vals else 0
    vmax = max(all_vals) if all_vals else 1
    vrange = vmax - vmin if vmax > vmin else 1

    svg = _svg_header(width, height)
    svg += _svg_text(width // 2, 25, title, size=16, weight="bold")

    offset_x = 150
    offset_y = 50

    # Draw cells
    for i, metric in enumerate(metrics):
        y = offset_y + i * cell_h
        svg += _svg_text(offset_x - 10, y + cell_h // 2 + 4, metric, size=9, anchor="end")

        for j, val in enumerate(layer_data[metric]):
            x = offset_x + j * cell_w
            # Color: blue (low) → white (mid) → red (high)
            norm_val = (val - vmin) / vrange
            r = int(255 * norm_val)
            b = int(255 * (1 - norm_val))
            g = int(100 * (1 - abs(norm_val - 0.5) * 2))
            color = f"#{r:02x}{g:02x}{b:02x}"
            svg += _svg_rect(x, y, cell_w - 1, cell_h - 1, fill=color)

            # Value text
            svg += _svg_text(
                x + cell_w // 2,
                y + cell_h // 2 + 4,
                f"{val:.2f}",
                size=7,
                color="black" if norm_val > 0.3 and norm_val < 0.7 else "white",
            )

    # Layer labels
    for j in range(n_layers):
        x = offset_x + j * cell_w + cell_w // 2
        svg += _svg_text(x, offset_y + total_h + 15, f"L{j}", size=9)

    svg += _svg_footer()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(svg)
    return svg


def bar_chart_with_errors(
    metric_names: list[str],
    jepa_means: list[float],
    jepa_cis: list[tuple[float, float]],
    baseline_means: list[float],
    baseline_cis: list[tuple[float, float]],
    title="JEPA vs Baseline",
    ylabel="Value",
    width=900,
    height=500,
    output_path=None,
):
    """Bar chart comparing JEPA vs baseline with error bars (CI).

    Args:
        metric_names: list of metric names
        jepa_means: JEPA mean values
        jepa_cis: [(lower, upper)] confidence intervals
        baseline_means: baseline mean values
        baseline_cis: [(lower, upper)] confidence intervals
        output_path: save SVG
    """
    n = len(metric_names)
    if n == 0:
        return None

    bar_width = 25
    group_width = 3 * bar_width + 10
    total_w = group_width * n + 100
    chart_w = max(total_w, width)
    chart_h = height - 100

    svg = _svg_header(chart_w, height)
    svg += _svg_text(chart_w // 2, 25, title, size=16, weight="bold")

    offset_x = 80
    offset_y = 50

    # Find y range
    all_vals = jepa_means + baseline_means + [ci[0] for ci in jepa_cis] + [ci[1] for ci in jepa_cis]
    y_min = min(all_vals) if all_vals else 0
    y_max = max(all_vals) if all_vals else 1
    y_range = y_max - y_min if y_max > y_min else 1

    def y_pos(val):
        return offset_y + chart_h * (1 - (val - y_min) / y_range)

    # Y axis
    svg += _svg_line(offset_x, offset_y, offset_x, offset_y + chart_h, color="black")
    svg += _svg_text(15, offset_y + chart_h // 2, ylabel, size=11, anchor="middle")

    # Draw bars
    for i, name in enumerate(metric_names):
        gx = offset_x + 20 + i * group_width

        # JEPA bar
        jh = chart_h * (jepa_means[i] - y_min) / y_range
        jy = offset_y + chart_h - jh
        svg += _svg_rect(gx, jy, bar_width, jh, fill="#4285f4")

        # JEPA error bar
        j_lo = y_pos(jepa_cis[i][0])
        j_hi = y_pos(jepa_cis[i][1])
        svg += _svg_line(
            gx + bar_width // 2, j_lo, gx + bar_width // 2, j_hi, color="#2a5db0", width=2
        )
        svg += _svg_line(
            gx + bar_width // 2 - 4, j_lo, gx + bar_width // 2 + 4, j_lo, color="#2a5db0", width=2
        )
        svg += _svg_line(
            gx + bar_width // 2 - 4, j_hi, gx + bar_width // 2 + 4, j_hi, color="#2a5db0", width=2
        )

        # Baseline bar
        bh = chart_h * (baseline_means[i] - y_min) / y_range
        by = offset_y + chart_h - bh
        svg += _svg_rect(gx + bar_width + 5, by, bar_width, bh, fill="#f4a142")

        # Baseline error bar
        b_lo = y_pos(baseline_cis[i][0])
        b_hi = y_pos(baseline_cis[i][1])
        svg += _svg_line(
            gx + bar_width + 5 + bar_width // 2,
            b_lo,
            gx + bar_width + 5 + bar_width // 2,
            b_hi,
            color="#b07020",
            width=2,
        )

        # Label
        svg += _svg_text(gx + bar_width + 2, offset_y + chart_h + 15, name, size=8)

    # Legend
    svg += _svg_rect(30, height - 40, 15, 15, fill="#4285f4")
    svg += _svg_text(55, height - 28, "JEPA", size=11, anchor="start")
    svg += _svg_rect(130, height - 40, 15, 15, fill="#f4a142")
    svg += _svg_text(155, height - 28, "Baseline", size=11, anchor="start")

    svg += _svg_footer()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(svg)
    return svg


def probing_complexity_curve(
    depths: list[int],
    jepa_accuracies: list[float],
    baseline_accuracies: list[float],
    task_name: str = "default",
    width=600,
    height=400,
    output_path=None,
):
    """Line plot showing probe accuracy vs probe depth for both models.

    The KEY FIGURE for the paper: if JEPA reaches target accuracy
    at lower depth, the gap between the curves IS the result.
    """
    svg = _svg_header(width, height)
    svg += _svg_text(width // 2, 25, f"Probing Complexity: {task_name}", size=14, weight="bold")

    margin = 60
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin - 30

    def x_pos(d):
        return margin + (d - min(depths)) / max(max(depths) - min(depths), 1) * chart_w

    def y_pos(a):
        return margin + chart_h * (1 - a)

    # Axes
    svg += _svg_line(margin, margin, margin, margin + chart_h, color="black")
    svg += _svg_line(margin, margin + chart_h, margin + chart_w, margin + chart_h, color="black")
    svg += _svg_text(15, margin + chart_h // 2, "Accuracy", size=10, anchor="middle")
    svg += _svg_text(margin + chart_w // 2, height - 10, "Probe Depth", size=10)

    # Grid lines
    for acc in [0.2, 0.4, 0.6, 0.8, 1.0]:
        y = y_pos(acc)
        svg += _svg_line(margin, y, margin + chart_w, y, color="#eeeeee", width=1)
        svg += _svg_text(margin - 10, y + 4, f"{acc:.1f}", size=8, anchor="end")

    for d in depths:
        x = x_pos(d)
        svg += _svg_text(x, margin + chart_h + 15, str(d), size=9)

    # JEPA line
    _n_j = min(len(depths), len(jepa_accuracies))
    jepa_points = sorted((x_pos(depths[i]), y_pos(jepa_accuracies[i])) for i in range(_n_j))
    if len(jepa_points) > 1:
        for i in range(len(jepa_points) - 1):
            svg += _svg_line(
                jepa_points[i][0],
                jepa_points[i][1],
                jepa_points[i + 1][0],
                jepa_points[i + 1][1],
                color="#4285f4",
                width=3,
            )
    for x, y in jepa_points:
        svg += _svg_circle(x, y, 4, fill="#4285f4")

    # Baseline line
    _n_b = min(len(depths), len(baseline_accuracies))
    base_points = sorted((x_pos(depths[i]), y_pos(baseline_accuracies[i])) for i in range(_n_b))
    if len(base_points) > 1:
        for i in range(len(base_points) - 1):
            svg += _svg_line(
                base_points[i][0],
                base_points[i][1],
                base_points[i + 1][0],
                base_points[i + 1][1],
                color="#f4a142",
                width=3,
            )
    for x, y in base_points:
        svg += _svg_circle(x, y, 4, fill="#f4a142")

    # Legend
    svg += _svg_line(width - 150, 50, width - 120, 50, color="#4285f4", width=3)
    svg += _svg_text(width - 115, 54, "JEPA", size=10, anchor="start")
    svg += _svg_line(width - 150, 70, width - 120, 70, color="#f4a142", width=3)
    svg += _svg_text(width - 115, 74, "Baseline", size=10, anchor="start")

    svg += _svg_footer()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(svg)
    return svg


def convergence_plot(
    steps: list[int],
    jepa_values: list[float],
    baseline_values: list[float],
    metric_name: str = "Loss",
    width=600,
    height=400,
    output_path=None,
):
    """Training convergence comparison plot."""
    svg = _svg_header(width, height)
    svg += _svg_text(width // 2, 25, f"Convergence: {metric_name}", size=14, weight="bold")

    margin = 60
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin - 30

    all_vals = list(jepa_values) + list(baseline_values)
    y_min = min(all_vals) if all_vals else 0
    y_max = max(all_vals) if all_vals else 1
    y_range = y_max - y_min if y_max > y_min else 1
    x_min, x_max = min(steps), max(steps)
    x_range = x_max - x_min if x_max > x_min else 1

    def xp(s):
        return margin + (s - x_min) / x_range * chart_w

    def yp(v):
        return margin + chart_h * (1 - (v - y_min) / y_range)

    svg += _svg_line(margin, margin, margin, margin + chart_h, color="black")
    svg += _svg_line(margin, margin + chart_h, margin + chart_w, margin + chart_h, color="black")

    # Values may lag steps in length; draw only fully-defined segments.
    n_j = max(min(len(steps), len(jepa_values)) - 1, 0)
    for i in range(n_j):
        svg += _svg_line(
            xp(steps[i]),
            yp(jepa_values[i]),
            xp(steps[i + 1]),
            yp(jepa_values[i + 1]),
            color="#4285f4",
            width=2,
        )

    n_b = max(min(len(steps), len(baseline_values)) - 1, 0)
    for i in range(n_b):
        svg += _svg_line(
            xp(steps[i]),
            yp(baseline_values[i]),
            xp(steps[i + 1]),
            yp(baseline_values[i + 1]),
            color="#f4a142",
            width=2,
        )

    svg += _svg_line(width - 150, 50, width - 120, 50, color="#4285f4", width=2)
    svg += _svg_text(width - 115, 54, "JEPA", size=10, anchor="start")
    svg += _svg_line(width - 150, 70, width - 120, 70, color="#f4a142", width=2)
    svg += _svg_text(width - 115, 74, "Baseline", size=10, anchor="start")

    svg += _svg_footer()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(svg)
    return svg


def ablation_comparison_chart(
    ablation_names,
    metric_name,
    values,
    full_model_value=None,
    title="Ablation Study",
    width=800,
    height=400,
    output_path=None,
):
    """Horizontal bar chart showing each ablation's impact on a metric."""
    n = len(ablation_names)
    if n == 0:
        return None

    bar_height = 25
    margin_top = 50
    margin_left = 150
    margin_right = 80
    total_h = margin_top + n * (bar_height + 5) + 40
    chart_w = width - margin_left - margin_right

    all_vals = list(values) + ([full_model_value] if full_model_value is not None else [])
    v_min = min(all_vals) if all_vals else 0
    v_max = max(all_vals) if all_vals else 1
    v_range = v_max - v_min if v_max > v_min else 1
    v_min -= v_range * 0.1
    v_max += v_range * 0.1
    v_range = v_max - v_min

    svg = _svg_header(width, total_h)
    svg += _svg_text(width // 2, 25, title, size=14, weight="bold")
    svg += _svg_text(width // 2, 42, f"Metric: {metric_name}", size=10, color="#666666")

    for i, (name, val) in enumerate(zip(ablation_names, values)):
        y = margin_top + i * (bar_height + 5)
        bar_w = max(chart_w * (val - v_min) / v_range, 2)
        if full_model_value is not None:
            color = "#e74c3c" if val < full_model_value else "#2ecc71"
        else:
            color = "#4285f4"
        svg += _svg_rect(margin_left, y, bar_w, bar_height, fill=color, opacity=0.7)
        svg += _svg_text(margin_left - 5, y + bar_height // 2 + 4, name, size=9, anchor="end")
        svg += _svg_text(
            margin_left + bar_w + 5, y + bar_height // 2 + 4, f"{val:.3f}", size=8, anchor="start"
        )

    if full_model_value is not None:
        x_line = margin_left + chart_w * (full_model_value - v_min) / v_range
        y_bottom = margin_top + n * (bar_height + 5)
        svg += f'  <line x1="{x_line}" y1="{margin_top - 5}" x2="{x_line}" y2="{y_bottom}" stroke="#333333" stroke-width="2" stroke-dasharray="5,5"/>\n'
        svg += _svg_text(x_line, y_bottom + 15, f"Full: {full_model_value:.3f}", size=8)

    svg += _svg_footer()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(svg)
    return svg


def scaling_law_plot(
    sizes,
    jepa_metrics,
    baseline_metrics,
    metric_name="Effective Dimension",
    jepa_fit=None,
    baseline_fit=None,
    width=700,
    height=450,
    output_path=None,
):
    """Log-log plot showing how metrics scale with model size."""

    svg = _svg_header(width, height)
    svg += _svg_text(width // 2, 20, f"Scaling: {metric_name}", size=14, weight="bold")

    margin = 70
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin - 30

    log_sizes = [math.log(max(s, 1)) for s in sizes]
    log_jepa = [math.log(max(m, 1e-10)) for m in jepa_metrics]
    log_base = [math.log(max(m, 1e-10)) for m in baseline_metrics]

    all_x = log_sizes
    all_y = log_jepa + log_base
    x_min_v, x_max_v = min(all_x), max(all_x)
    y_min_v, y_max_v = min(all_y), max(all_y)
    x_rng = max(x_max_v - x_min_v, 1)
    y_rng = max(y_max_v - y_min_v, 1)
    x_min_v -= x_rng * 0.05
    x_max_v += x_rng * 0.05
    y_min_v -= y_rng * 0.05
    y_max_v += y_rng * 0.05
    x_rng = x_max_v - x_min_v
    y_rng = y_max_v - y_min_v

    def xp(v):
        return margin + (v - x_min_v) / x_rng * chart_w

    def yp(v):
        return margin + chart_h * (1 - (v - y_min_v) / y_rng)

    svg += _svg_line(margin, margin, margin, margin + chart_h, color="black")
    svg += _svg_line(margin, margin + chart_h, margin + chart_w, margin + chart_h, color="black")
    svg += _svg_text(15, margin + chart_h // 2, f"log({metric_name})", size=9, anchor="middle")
    svg += _svg_text(margin + chart_w // 2, height - 8, "log(Params)", size=9)

    for frac in [0.25, 0.5, 0.75]:
        gx = margin + chart_w * frac
        gy = margin + chart_h * frac
        svg += _svg_line(gx, margin, gx, margin + chart_h, color="#eeeeee", width=1)
        svg += _svg_line(margin, gy, margin + chart_w, gy, color="#eeeeee", width=1)

    for i in range(len(log_sizes)):
        cx, cy = xp(log_sizes[i]), yp(log_jepa[i])
        svg += _svg_circle(cx, cy, 5, fill="#4285f4")
        if i > 0:
            px, py = xp(log_sizes[i - 1]), yp(log_jepa[i - 1])
            svg += _svg_line(px, py, cx, cy, color="#4285f4", width=2)

    for i in range(len(log_sizes)):
        cx, cy = xp(log_sizes[i]), yp(log_base[i])
        svg += _svg_circle(cx, cy, 5, fill="#f4a142")
        if i > 0:
            px, py = xp(log_sizes[i - 1]), yp(log_base[i - 1])
            svg += _svg_line(px, py, cx, cy, color="#f4a142", width=2)

    svg += _svg_line(width - 150, 50, width - 120, 50, color="#4285f4", width=3)
    svg += _svg_text(width - 115, 54, "JEPA", size=10, anchor="start")
    svg += _svg_line(width - 150, 70, width - 120, 70, color="#f4a142", width=3)
    svg += _svg_text(width - 115, 74, "Baseline", size=10, anchor="start")

    svg += _svg_footer()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(svg)
    return svg


def robustness_curve(
    intensities,
    jepa_cka,
    baseline_cka,
    perturbation_name="Token Dropout",
    width=600,
    height=400,
    output_path=None,
):
    """Line plot showing CKA degradation under increasing perturbation."""
    svg = _svg_header(width, height)
    svg += _svg_text(width // 2, 20, f"Robustness: {perturbation_name}", size=14, weight="bold")

    margin = 60
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin - 30

    all_vals = jepa_cka + baseline_cka
    y_min_v = min(min(all_vals), 0)
    y_max_v = max(max(all_vals), 1)
    y_rng = y_max_v - y_min_v if y_max_v > y_min_v else 1
    x_min_v, x_max_v = min(intensities), max(intensities)
    x_rng = x_max_v - x_min_v if x_max_v > x_min_v else 1

    def xp(v):
        return margin + (v - x_min_v) / x_rng * chart_w

    def yp(v):
        return margin + chart_h * (1 - (v - y_min_v) / y_rng)

    svg += _svg_line(margin, margin, margin, margin + chart_h, color="black")
    svg += _svg_line(margin, margin + chart_h, margin + chart_w, margin + chart_h, color="black")
    svg += _svg_text(15, margin + chart_h // 2, "CKA(clean, perturbed)", size=9, anchor="middle")
    svg += _svg_text(margin + chart_w // 2, height - 8, "Perturbation Intensity", size=9)

    for acc in [0.2, 0.4, 0.6, 0.8, 1.0]:
        y = yp(acc)
        svg += _svg_line(margin, y, margin + chart_w, y, color="#eeeeee", width=1)
        svg += _svg_text(margin - 5, y + 4, f"{acc:.1f}", size=8, anchor="end")

    for i in range(len(intensities) - 1):
        svg += _svg_line(
            xp(intensities[i]),
            yp(jepa_cka[i]),
            xp(intensities[i + 1]),
            yp(jepa_cka[i + 1]),
            color="#4285f4",
            width=3,
        )
    for x, y in zip(intensities, jepa_cka):
        svg += _svg_circle(xp(x), yp(y), 4, fill="#4285f4")

    for i in range(len(intensities) - 1):
        svg += _svg_line(
            xp(intensities[i]),
            yp(baseline_cka[i]),
            xp(intensities[i + 1]),
            yp(baseline_cka[i + 1]),
            color="#f4a142",
            width=3,
        )
    for x, y in zip(intensities, baseline_cka):
        svg += _svg_circle(xp(x), yp(y), 4, fill="#f4a142")

    if len(intensities) > 1:
        for i in range(len(intensities) - 1):
            x1, x2 = xp(intensities[i]), xp(intensities[i + 1])
            jy1, jy2 = yp(jepa_cka[i]), yp(jepa_cka[i + 1])
            by1, by2 = yp(baseline_cka[i]), yp(baseline_cka[i + 1])
            pts = f"{x1},{jy1} {x2},{jy2} {x2},{by2} {x1},{by1}"
            svg += f'  <polygon points="{pts}" fill="#4285f4" opacity="0.1"/>\n'

    svg += _svg_line(width - 150, 50, width - 120, 50, color="#4285f4", width=3)
    svg += _svg_text(width - 115, 54, "JEPA", size=10, anchor="start")
    svg += _svg_line(width - 150, 70, width - 120, 70, color="#f4a142", width=3)
    svg += _svg_text(width - 115, 74, "Baseline", size=10, anchor="start")

    svg += _svg_footer()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(svg)
    return svg


def information_plane(
    mi_input,
    mi_task,
    layer_labels=None,
    title="Information Plane",
    width=600,
    height=500,
    output_path=None,
):
    """Information plane: MI(input) vs MI(task) across layers."""
    svg = _svg_header(width, height)
    svg += _svg_text(width // 2, 20, title, size=14, weight="bold")

    margin = 70
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin - 20

    x_min_v = min(mi_input) - 0.1
    x_max_v = max(mi_input) + 0.1
    y_min_v = min(mi_task) - 0.1
    y_max_v = max(mi_task) + 0.1
    x_rng = max(x_max_v - x_min_v, 0.1)
    y_rng = max(y_max_v - y_min_v, 0.1)

    def xp(v):
        return margin + (v - x_min_v) / x_rng * chart_w

    def yp(v):
        return margin + chart_h * (1 - (v - y_min_v) / y_rng)

    svg += _svg_line(margin, margin, margin, margin + chart_h, color="black")
    svg += _svg_line(margin, margin + chart_h, margin + chart_w, margin + chart_h, color="black")
    svg += _svg_text(15, margin + chart_h // 2, "MI(Y; Z)", size=10, anchor="middle")
    svg += _svg_text(margin + chart_w // 2, height - 5, "MI(X; Z)", size=10)

    colors = ["#ff6b6b", "#ffa502", "#7bed9f", "#70a1ff", "#5352ed", "#2ed573"]
    for i in range(len(mi_input) - 1):
        color = colors[i % len(colors)]
        svg += _svg_line(
            xp(mi_input[i]),
            yp(mi_task[i]),
            xp(mi_input[i + 1]),
            yp(mi_task[i + 1]),
            color=color,
            width=2,
        )

    for i, (x, y) in enumerate(zip(mi_input, mi_task)):
        color = colors[i % len(colors)]
        svg += _svg_circle(xp(x), yp(y), 5, fill=color)
        label = layer_labels[i] if layer_labels else f"L{i}"
        svg += _svg_text(xp(x) + 8, yp(y) - 5, label, size=8, anchor="start")

    svg += _svg_footer()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(svg)
    return svg
