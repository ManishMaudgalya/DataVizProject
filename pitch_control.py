
# pitch_control.py  —  Voronoi Pitch Control + Team Shape (Plotly)

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pass_network import (
    _base_pitch_figure, _pitch_shapes,
    PITCH_BG, PITCH_LINES, CYAN, PINK, GOLD, WHITE, TEXT_COLOR, DIM,
)

PITCH_W = 120.0
PITCH_H = 80.0


# VORONOI
def _rasterize_voronoi(freeze_frame: list, res: int = 2) -> np.ndarray:
    """
    Rasterize pitch into a team-ownership grid via nearest-player distance.
    Returns team_grid shape (H*res, W*res):  +1 = attacking, -1 = defending
    """
    if not freeze_frame:
        return None

    positions   = np.array([[p["x"], p["y"]] for p in freeze_frame], dtype=float)
    is_teammate = np.array([1 if p["teammate"] else -1 for p in freeze_frame])

    n_x = int(PITCH_W * res)   # ← CRITICAL int() cast
    n_y = int(PITCH_H * res)   # ← CRITICAL int() cast

    gx, gy = np.meshgrid(
        np.linspace(0.0, PITCH_W, n_x),
        np.linspace(0.0, PITCH_H, n_y),
    )
    grid_pts = np.c_[gx.ravel(), gy.ravel()]

    # Vectorized nearest-player: (n_grid, n_players) → argmin along player axis
    dists   = np.sqrt(((grid_pts[:, None, :] - positions[None, :, :]) ** 2).sum(2))
    nearest = np.argmin(dists, axis=1)
    return is_teammate[nearest].reshape(gy.shape)



# PITCH CONTROL 
def plot_pitch_control(
    freeze_frame: list,
    event_description: str = "",
    match_label: str = "",
    team_a_name: str = "Attacking team",
    team_b_name: str = "Defending team",
) -> go.Figure:
    """
    Layers
      1. Heatmap — team-ownership colour fill
      2. Glowing player dots with hover cards
    """
    fig = _base_pitch_figure(height=600)

    if not freeze_frame:
        fig.add_annotation(x=60, y=40,
            text="No freeze-frame data available for this event.",
            font=dict(color=CYAN, size=14, family="Courier New"),
            showarrow=False)
        return fig

    # Ownership grid
    res  = 2
    grid = _rasterize_voronoi(freeze_frame, res=res)

    if grid is not None:
        n_x = int(PITCH_W * res)
        n_y = int(PITCH_H * res)
        # Normalise −1/+1 → 0/1 for heatmap colorscale
        grid_norm = (grid.astype(float) + 1.0) / 2.0

        fig.add_trace(go.Heatmap(
            z=grid_norm,
            x=np.linspace(0, PITCH_W, n_x),
            y=np.linspace(0, PITCH_H, n_y),
            colorscale=[
                [0.0,  "rgba(255,  0,255,0.45)"],
                [0.35, "rgba(255,  0,255,0.12)"],
                [0.5,  "rgba( 11, 12, 16,0.00)"],
                [0.65, "rgba(102,252,241,0.12)"],
                [1.0,  "rgba(102,252,241,0.45)"],
            ],
            showscale=False,
            hoverinfo="none",
            zmin=0, zmax=1,
        ))

        atk_pct = float((grid == 1).sum()) / grid.size * 100
        def_pct = 100.0 - atk_pct
    else:
        atk_pct = def_pct = 50.0

    # Player traces
    for players, color, label in [
        ([p for p in freeze_frame if not p["teammate"]], PINK,  team_b_name),
        ([p for p in freeze_frame if p["teammate"]],     CYAN,  team_a_name),
    ]:
        if not players:
            continue

        # Outer glow
        for gi in (3, 2):
            fig.add_trace(go.Scatter(
                x=[p["x"] for p in players], y=[p["y"] for p in players],
                mode="markers",
                marker=dict(size=30 + gi * 10, color=color,
                            opacity=0.08 / gi, line=dict(width=0)),
                showlegend=False, hoverinfo="none",
            ))

        # Main dots
        sizes = [24 if p.get("keeper") else 18 for p in players]
        hover = [
            f"<b>{'GK — ' if p.get('keeper') else ''}{label}</b><br>"
            f"Position: ({p['x']:.1f}, {p['y']:.1f})<br>"
            + ("⚽ Ball holder" if p.get("actor") else "")
            for p in players
        ]
        fig.add_trace(go.Scatter(
            x=[p["x"] for p in players], y=[p["y"] for p in players],
            mode="markers",
            marker=dict(size=sizes, color=color, opacity=1.0,
                        symbol=["diamond" if p.get("keeper") else "circle"
                                for p in players],
                        line=dict(color=WHITE, width=2)),
            name=f"{label} ({atk_pct:.0f}% space)" if color == CYAN
                 else f"{label} ({def_pct:.0f}% space)",
            hovertext=hover, hoverinfo="text",
        ))

        # Gold diamond for ball actor
        actors = [p for p in players if p.get("actor")]
        if actors:
            fig.add_trace(go.Scatter(
                x=[p["x"] for p in actors], y=[p["y"] for p in actors],
                mode="markers",
                marker=dict(size=22, color=GOLD, symbol="diamond",
                            line=dict(color=WHITE, width=2)),
                name="Ball holder",
                hovertext=[f"<b>Ball holder</b><br>({p['x']:.1f}, {p['y']:.1f})"
                           for p in actors],
                hoverinfo="text",
            ))

    # Space % annotation
    fig.add_annotation(
        x=2, y=3,
        text=f"■ {team_a_name}: {atk_pct:.0f}%  &nbsp; ■ {team_b_name}: {def_pct:.0f}%",
        font=dict(color=WHITE, size=11, family="Courier New"),
        showarrow=False, xanchor="left",
    )
    if event_description:
        fig.add_annotation(
            x=60, y=77, text=event_description,
            font=dict(color=CYAN, size=10, family="Courier New"),
            showarrow=False, xanchor="center",
        )

    return fig


# TEAM SHAPE — single team
def plot_team_shape_single(
    df: pd.DataFrame,
    team_name: str,
    minute_min: int = 0,
    minute_max: int = 130,
    color: str = PINK,
    height: int = 560,
) -> go.Figure:
    fig = _base_pitch_figure(height=height)

    team_df = df[
        (df["team_name"] == team_name) &
        df["x"].notna() & df["y"].notna() &
        df["minute"].between(minute_min, minute_max) &
        df["player_name"].notna() & (df["player_name"] != "")
    ].copy()

    if team_df.empty:
        fig.add_annotation(x=60, y=40,
            text=f"No position data for {team_name} in this time window.",
            font=dict(color=CYAN, size=13, family="Courier New"),
            showarrow=False)
        return fig

    stats = []
    for pname, grp in team_df.groupby("player_name"):
        stats.append({
            "player_name": pname,
            "avg_x"      : float(grp["x"].mean()),
            "avg_y"      : float(grp["y"].mean()),
            "n_events"   : len(grp),
            "n_passes"   : int((grp["event_type"] == "Pass").sum()),
            "n_shots"    : int((grp["event_type"] == "Shot").sum()),
        })
    stats_df = pd.DataFrame(stats)

    mx   = float(stats_df["n_events"].max())
    mn   = float(stats_df["n_events"].min())
    rng  = max(mx - mn, 1.0)
    sizes = [16 + 22 * (r.n_events - mn) / rng for _, r in stats_df.iterrows()]

    # Glow
    for gi in (3, 2, 1):
        fig.add_trace(go.Scatter(
            x=stats_df["avg_x"].tolist(), y=stats_df["avg_y"].tolist(),
            mode="markers",
            marker=dict(size=[s + gi * 12 for s in sizes], color=color,
                        opacity=0.09 / gi, line=dict(width=0)),
            showlegend=False, hoverinfo="none",
        ))

    last_names  = [n.split()[-1] for n in stats_df["player_name"]]
    hover_texts = [
        f"<b>{r.player_name}</b><br>"
        f"Avg position: ({r.avg_x:.1f}, {r.avg_y:.1f})<br>"
        f"Total events: {r.n_events}<br>"
        f"Passes: {r.n_passes}  Shots: {r.n_shots}"
        for _, r in stats_df.iterrows()
    ]

    fig.add_trace(go.Scatter(
        x=stats_df["avg_x"].tolist(),
        y=stats_df["avg_y"].tolist(),
        mode="markers+text",
        marker=dict(size=sizes, color=color, opacity=1.0,
                    line=dict(color=WHITE, width=2)),
        text=last_names,
        textposition="bottom center",
        textfont=dict(color=CYAN, size=9, family="Courier New"),
        hovertext=hover_texts,
        hoverinfo="text",
        name=f"{team_name} players",
    ))

    # Title annotation INSIDE the figure (so each side-by-side pitch is labelled)
    fig.add_annotation(
        x=60, y=77, text=f"<b>{team_name}</b>",
        font=dict(color=color, size=14, family="Courier New"),
        showarrow=False, xanchor="center",
    )

    return fig



# TEAM SHAPE — both teams (returns TWO figures for side-by-side rendering)
def plot_team_shape_both(
    df: pd.DataFrame,
    team_a: str,
    team_b: str,
    minute_min: int = 0,
    minute_max: int = 130,
) -> tuple:
    """
    Build TWO independent team-shape figures — one per team — for the caller
    to render side-by-side via st.columns.

    Returns (fig_a, fig_b).
    """
    fig_a = plot_team_shape_single(
        df, team_a, minute_min=minute_min, minute_max=minute_max,
        color=CYAN, height=480,
    )
    fig_b = plot_team_shape_single(
        df, team_b, minute_min=minute_min, minute_max=minute_max,
        color=PINK, height=480,
    )
    return fig_a, fig_b
