# =============================================================================
# pass_network.py  — Interactive Pass Network (Plotly)
#
# ROOT-CAUSE FIX: All pitch shapes now carry layer="below" so they render
# BEHIND scatter traces. Previously the filled background rect defaulted to
# layer="above", painting over every marker and line — making the network
# "invisible" while hover still fired (because hover uses the data geometry,
# not the visual layer).
# =============================================================================

import numpy as np
import pandas as pd
from collections import defaultdict
import plotly.graph_objects as go

# ── Theme constants ──────────────────────────────────────────────────────────
PITCH_BG    = "#0b0c10"
PITCH_LINES = "#1f2833"
CYAN        = "#66fcf1"
PINK        = "#ff00ff"
GOLD        = "#f5c518"
WHITE       = "#ffffff"
TEXT_COLOR  = "#c5c6c7"
DIM         = "#445566"


# ============================================================================
# PITCH HELPERS
# ============================================================================

def _pitch_shapes(lc: str = PITCH_LINES, lw: float = 1.5) -> list:
    """
    Plotly layout shapes for a StatsBomb 120×80-yard pitch.

    EVERY shape has layer='below' — without this, the filled background rect
    renders above all scatter traces and hides every visible mark.
    """
    ls = dict(color=lc, width=lw)
    nf = "rgba(0,0,0,0)"
    BL = "below"

    return [
        # Background fill — must be below or it hides everything
        dict(type="rect",   x0=0,     y0=0,    x1=120,   y1=80,
             fillcolor=PITCH_BG, line=dict(color=lc, width=0), layer=BL),
        # Pitch outline
        dict(type="rect",   x0=0,     y0=0,    x1=120,   y1=80,
             fillcolor=nf,  line=ls,  layer=BL),
        # Halfway line
        dict(type="line",   x0=60,    y0=0,    x1=60,    y1=80, line=ls, layer=BL),
        # Centre circle
        dict(type="circle", x0=50,    y0=30,   x1=70,    y1=50,
             fillcolor=nf,  line=ls,  layer=BL),
        # Centre spot
        dict(type="circle", x0=59.6,  y0=39.6, x1=60.4,  y1=40.4,
             fillcolor=lc,  line=dict(color=lc, width=0), layer=BL),
        # Left penalty area
        dict(type="rect",   x0=0,     y0=18,   x1=18,    y1=62,
             fillcolor=nf,  line=ls,  layer=BL),
        # Right penalty area
        dict(type="rect",   x0=102,   y0=18,   x1=120,   y1=62,
             fillcolor=nf,  line=ls,  layer=BL),
        # Left 6-yard box
        dict(type="rect",   x0=0,     y0=30,   x1=6,     y1=50,
             fillcolor=nf,  line=ls,  layer=BL),
        # Right 6-yard box
        dict(type="rect",   x0=114,   y0=30,   x1=120,   y1=50,
             fillcolor=nf,  line=ls,  layer=BL),
        # Penalty spots
        dict(type="circle", x0=11.4,  y0=39.4, x1=12.6,  y1=40.6,
             fillcolor=lc,  line=dict(color=lc, width=0), layer=BL),
        dict(type="circle", x0=107.4, y0=39.4, x1=108.6, y1=40.6,
             fillcolor=lc,  line=dict(color=lc, width=0), layer=BL),
        # Goals
        dict(type="rect",   x0=-2,    y0=36,   x1=0,     y1=44,
             fillcolor="rgba(31,40,51,0.8)", line=ls, layer=BL),
        dict(type="rect",   x0=120,   y0=36,   x1=122,   y1=44,
             fillcolor="rgba(31,40,51,0.8)", line=ls, layer=BL),
    ]


def _base_pitch_figure(title: str = "", height: int = 580) -> go.Figure:
    """
    Plotly Figure pre-configured with a dark Cyberpunk pitch.

    scaleanchor is intentionally omitted — it collapses Streamlit column
    widths and distorts coordinate mapping in narrow containers.
    """
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=PITCH_BG,
        plot_bgcolor=PITCH_BG,
        height=height,
        margin=dict(l=5, r=5, t=10, b=10),
        shapes=_pitch_shapes(),
        xaxis=dict(
            range=[-3, 123],
            showgrid=False, zeroline=False,
            showticklabels=False, fixedrange=True,
        ),
        yaxis=dict(
            range=[-3, 83],
            showgrid=False, zeroline=False,
            showticklabels=False, fixedrange=True,
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(11,12,16,0.88)", bordercolor=PITCH_LINES,
            borderwidth=1, font=dict(color=TEXT_COLOR, size=11),
            x=0.01, y=0.01, orientation="h",
        ),
        hoverlabel=dict(
            bgcolor="#1a2332", bordercolor=CYAN,
            font=dict(color=WHITE, size=12, family="Courier New"),
        ),
        dragmode=False,
    )
    return fig


# ============================================================================
# BUILD PASS NETWORK
# ============================================================================

def build_pass_network(
    df: pd.DataFrame,
    team_name: str,
    minute_min: int = 0,
    minute_max: int = 130,
    min_passes: int = 2,
) -> tuple:
    """
    Extract pass-network nodes and edges from a match events DataFrame.

    Returns (nodes_df, edges_df, best_partners_dict).
    """
    mask = (
        (df["team_name"] == team_name) &
        df["minute"].between(minute_min, minute_max) &
        (df["event_type"] == "Pass") &
        (df["result"] == "Complete") &
        df["x"].notna() & df["y"].notna()
    )
    passes = df[mask].copy()

    if passes.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # Nodes: average position per passer
    node_rows = []
    for pid, grp in passes.groupby("player_id"):
        valid = grp.dropna(subset=["x", "y"])
        if len(valid) < min_passes:
            continue
        node_rows.append({
            "player_id"  : str(pid),
            "player_name": grp["player_name"].iloc[0],
            "avg_x"      : float(valid["x"].mean()),
            "avg_y"      : float(valid["y"].mean()),
            "pass_count" : len(valid),
        })

    if not node_rows:
        return pd.DataFrame(), pd.DataFrame(), {}

    nodes_df   = pd.DataFrame(node_rows)
    valid_pids = set(nodes_df["player_id"])
    name_map   = dict(zip(nodes_df["player_id"], nodes_df["player_name"]))

    # Edges: pass counts between pairs
    edge_counts = defaultdict(int)
    partner_map = defaultdict(lambda: defaultdict(int))

    for _, row in passes.iterrows():
        passer = str(row["player_id"])
        recip  = str(row.get("pass_recipient_id", ""))
        if not recip or passer not in valid_pids or recip not in valid_pids:
            continue
        key = tuple(sorted([passer, recip]))
        edge_counts[key]           += 1
        partner_map[passer][recip] += 1
        partner_map[recip][passer] += 1

    edge_rows = [
        {"player_a_id": a, "player_b_id": b,
         "player_a_name": name_map.get(a, a),
         "player_b_name": name_map.get(b, b),
         "pass_count": cnt}
        for (a, b), cnt in edge_counts.items()
        if cnt >= min_passes
    ]
    edges_df = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(
        columns=["player_a_id","player_b_id","player_a_name","player_b_name","pass_count"])

    best_partners = {}
    for pid, partners in partner_map.items():
        if partners:
            best_id = max(partners, key=partners.get)
            best_partners[pid] = (name_map.get(best_id, best_id), partners[best_id])

    return nodes_df, edges_df, best_partners


# ============================================================================
# RENDER PASS NETWORK
# ============================================================================

def plot_pass_network(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    best_partners: dict,
    team_name: str,
    match_label: str = "",
    minute_label: str = "",
) -> go.Figure:
    """
    Render an interactive Cyberpunk pass network on a Plotly pitch.
    Player nodes are PINK (size ∝ pass volume), edges are CYAN (width ∝ frequency).
    """
    fig = _base_pitch_figure(height=600)

    if nodes_df.empty:
        fig.add_annotation(x=60, y=40,
            text="No completed passes found — try lowering the min-passes threshold.",
            font=dict(color=CYAN, size=13, family="Courier New"), showarrow=False)
        return fig

    node_pos = nodes_df.set_index("player_id")[["avg_x","avg_y"]].to_dict("index")

    # ── EDGES ──────────────────────────────────────────────────────────
    if not edges_df.empty:
        max_p = float(edges_df["pass_count"].max())
        min_p = float(edges_df["pass_count"].min())
        rng   = max(max_p - min_p, 1.0)

        for _, edge in edges_df.sort_values("pass_count").iterrows():
            pa = node_pos.get(edge.player_a_id)
            pb = node_pos.get(edge.player_b_id)
            if not pa or not pb:
                continue

            norm    = (edge.pass_count - min_p) / rng
            lw_base = 1.5 + 5.5 * norm
            alpha   = 0.40 + 0.60 * norm

            # Glow rings (3 layers, widening + fading)
            for gi in (3, 2, 1):
                fig.add_trace(go.Scatter(
                    x=[pa["avg_x"], pb["avg_x"], None],
                    y=[pa["avg_y"], pb["avg_y"], None],
                    mode="lines",
                    line=dict(color=CYAN, width=lw_base * (1 + gi * 2.2)),
                    opacity=alpha * 0.10 / gi,
                    showlegend=False, hoverinfo="none",
                ))

            # Main edge
            fig.add_trace(go.Scatter(
                x=[pa["avg_x"], pb["avg_x"], None],
                y=[pa["avg_y"], pb["avg_y"], None],
                mode="lines",
                line=dict(color=CYAN, width=lw_base),
                opacity=alpha,
                showlegend=False, hoverinfo="none",
            ))

            # Hover hit-box at midpoint
            fig.add_trace(go.Scatter(
                x=[(pa["avg_x"] + pb["avg_x"]) / 2],
                y=[(pa["avg_y"] + pb["avg_y"]) / 2],
                mode="markers",
                marker=dict(size=12, color="rgba(0,0,0,0)"),
                customdata=[[edge.player_a_name, edge.player_b_name, int(edge.pass_count)]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> ↔ <b>%{customdata[1]}</b><br>"
                    "Passes: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            ))

    # ── NODES ──────────────────────────────────────────────────────────
    max_n = float(nodes_df["pass_count"].max())
    min_n = float(nodes_df["pass_count"].min())
    n_rng = max(max_n - min_n, 1.0)

    # Glow halos
    for gi in (3, 2, 1):
        fig.add_trace(go.Scatter(
            x=nodes_df["avg_x"].tolist(),
            y=nodes_df["avg_y"].tolist(),
            mode="markers",
            marker=dict(
                size=[24 + 28 * (r.pass_count - min_n) / n_rng + gi * 14
                      for _, r in nodes_df.iterrows()],
                color=PINK, opacity=0.12 / gi, line=dict(width=0)),
            showlegend=False, hoverinfo="none",
        ))

    # Main nodes
    node_sizes  = [20 + 24 * (r.pass_count - min_n) / n_rng for _, r in nodes_df.iterrows()]
    last_names  = [str(r.player_name).split()[-1] for _, r in nodes_df.iterrows()]
    hover_texts = [
        (
            f"<b>{r.player_name}</b><br>"
            f"Passes: {r.pass_count}<br>"
            f"Avg position: ({r.avg_x:.1f}, {r.avg_y:.1f})<br>"
            + (f"Top partner: <b>{best_partners[r.player_id][0]}</b> "
               f"({best_partners[r.player_id][1]} passes)"
               if r.player_id in best_partners else "")
        )
        for _, r in nodes_df.iterrows()
    ]

    fig.add_trace(go.Scatter(
        x=nodes_df["avg_x"].tolist(),
        y=nodes_df["avg_y"].tolist(),
        mode="markers+text",
        marker=dict(size=node_sizes, color=PINK, opacity=1.0,
                    line=dict(color=WHITE, width=2.0)),
        text=last_names,
        textposition="bottom center",
        textfont=dict(color=CYAN, size=9, family="Courier New"),
        hovertext=hover_texts,
        hoverinfo="text",
        name=f"{team_name} players",
    ))

    return fig