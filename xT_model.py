# =============================================================================
# xT_model.py  —  Expected Threat (12×8 Markov Chain)
#
# FIX: compute_threat_added was filtering by df["team_id"] but the caller
#      passes a team NAME string (e.g. "Germany").  team_id is a numeric
#      string like "944" — the comparison always failed → empty DataFrame →
#      "No threat data available".  Fixed to use df["team_name"].
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize

try:
    from pass_network import (
        PITCH_BG, LINE_COLOR, CYAN_NEON, PINK_NEON,
        GOLD_NEON, WHITE_GLOW, _make_pitch,
    )
except Exception:
    PITCH_BG   = "#0b0c10"
    LINE_COLOR = "#1f2833"
    CYAN_NEON  = "#66fcf1"
    PINK_NEON  = "#ff00ff"
    GOLD_NEON  = "#f5c518"
    WHITE_GLOW = "#ffffff"

    def _make_pitch():
        from mplsoccer import Pitch
        pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_BG,
                      line_color=LINE_COLOR, linewidth=1.0,
                      corner_arcs=True, goal_type="box")
        fig, ax = pitch.draw(figsize=(12, 8))
        fig.patch.set_facecolor(PITCH_BG)
        return fig, ax, pitch

# Alias so xT_model works standalone
LINE_COLOR = "#1f2833"
CYAN_NEON  = "#66fcf1"
PINK_NEON  = "#ff00ff"
GOLD_NEON  = "#f5c518"
WHITE_GLOW = "#ffffff"

# ── Grid constants ───────────────────────────────────────────────────────────
XT_COLS = 12
XT_ROWS = 8
PITCH_W = 120.0
PITCH_H = 80.0


# ============================================================================
# COORDINATE HELPERS
# ============================================================================

def coords_to_grid(x: float, y: float) -> tuple:
    col = int(np.clip(int(x / PITCH_W * XT_COLS), 0, XT_COLS - 1))
    row = int(np.clip(int(y / PITCH_H * XT_ROWS), 0, XT_ROWS - 1))
    return col, row


def grid_to_center(col: int, row: int) -> tuple:
    cw = PITCH_W / XT_COLS
    ch = PITCH_H / XT_ROWS
    return col * cw + cw / 2.0, row * ch + ch / 2.0


# ============================================================================
# BUILD TRANSITION MATRIX
# ============================================================================

def build_transition_matrix(df: pd.DataFrame) -> tuple:
    """
    Estimate T (12,8,12,8), p_shoot (12,8), p_goal (12,8) from event data.
    """
    T_counts      = np.zeros((XT_COLS, XT_ROWS, XT_COLS, XT_ROWS), dtype=np.float64)
    move_counts   = np.zeros((XT_COLS, XT_ROWS), dtype=np.float64)
    shoot_counts  = np.zeros((XT_COLS, XT_ROWS), dtype=np.float64)
    goal_counts   = np.zeros((XT_COLS, XT_ROWS), dtype=np.float64)
    action_counts = np.zeros((XT_COLS, XT_ROWS), dtype=np.float64)

    et  = df["event_type"].astype(str).str.lower()
    res = df["result"].astype(str).str.lower() if "result" in df.columns \
          else pd.Series([""] * len(df), index=df.index)

    def sf(v):
        try: return float(v)
        except: return np.nan

    # Completed passes
    pm = (et == "pass") & (res == "complete") & df["x"].notna() & df["end_x"].notna()
    for _, r in df[pm].iterrows():
        x1,y1,x2,y2 = sf(r["x"]),sf(r["y"]),sf(r["end_x"]),sf(r["end_y"])
        if any(np.isnan([x1,y1,x2,y2])): continue
        c1,r1 = coords_to_grid(x1,y1); c2,r2 = coords_to_grid(x2,y2)
        T_counts[c1,r1,c2,r2] += 1; move_counts[c1,r1] += 1; action_counts[c1,r1] += 1

    # Carries / dribbles
    cm = et.isin(["carry","dribble"]) & df["x"].notna() & df["end_x"].notna()
    for _, r in df[cm].iterrows():
        x1,y1,x2,y2 = sf(r["x"]),sf(r["y"]),sf(r["end_x"]),sf(r["end_y"])
        if any(np.isnan([x1,y1,x2,y2])): continue
        c1,r1 = coords_to_grid(x1,y1); c2,r2 = coords_to_grid(x2,y2)
        T_counts[c1,r1,c2,r2] += 1; move_counts[c1,r1] += 1; action_counts[c1,r1] += 1

    # Shots
    sm = (et == "shot") & df["x"].notna()
    for _, r in df[sm].iterrows():
        x1,y1 = sf(r["x"]),sf(r["y"])
        if np.isnan(x1) or np.isnan(y1): continue
        c,ro = coords_to_grid(x1,y1)
        shoot_counts[c,ro] += 1; action_counts[c,ro] += 1
        if str(r.get("result","")).lower() in ("goal","success"):
            goal_counts[c,ro] += 1

    p_shoot = np.clip(shoot_counts / np.maximum(action_counts, 1e-9), 0, 1)
    p_goal_raw = goal_counts / np.maximum(shoot_counts, 1e-9)

    p_goal = np.zeros((XT_COLS, XT_ROWS))
    for c in range(XT_COLS):
        for ro in range(XT_ROWS):
            cx,cy = grid_to_center(c, ro)
            dist  = np.sqrt((cx - PITCH_W)**2 + (cy - PITCH_H/2)**2)
            geo   = 1.0 / (1.0 + 0.05 * max(dist, 1.0)**1.7)
            w     = min(float(shoot_counts[c,ro]) / 20.0, 1.0)
            p_goal[c,ro] = w * p_goal_raw[c,ro] + (1 - w) * geo

    T = T_counts / np.maximum(move_counts[:,:,None,None], 1e-9)
    return T, p_shoot, p_goal


# ============================================================================
# VALUE ITERATION SOLVER
# ============================================================================

def solve_xt(T, p_shoot, p_goal, n_iter=60, convergence_threshold=1e-5):
    """Solve Bellman equation: xT(z) = P(shoot)*P(goal) + P(move)*Σ T*xT """
    xT     = np.zeros((XT_COLS, XT_ROWS))
    p_move = 1.0 - p_shoot
    T_flat = T.reshape(XT_COLS * XT_ROWS, XT_COLS * XT_ROWS)

    for i in range(n_iter):
        xT_old = xT.copy()
        move_val = (T_flat @ xT.flatten()).reshape(XT_COLS, XT_ROWS)
        xT = p_shoot * p_goal + p_move * move_val
        if np.max(np.abs(xT - xT_old)) < convergence_threshold:
            break
    return xT


# ============================================================================
# COMPUTE PER-ACTION THREAT ADDED
# ============================================================================

def compute_threat_added(df: pd.DataFrame, xT_grid: np.ndarray,
                         team_name: str) -> pd.DataFrame:
    """
    For each pass/carry by the team, compute xT_added = xT(end) - xT(start).

    FIX: was filtering by df["team_id"] but callers pass a team NAME string.
    team_id is a numeric string like "944"; "Germany" != "944" → always empty.
    Now correctly uses df["team_name"].
    """
    et = df["event_type"].astype(str).str.lower()

    # ── KEY FIX: use team_name, not team_id ──────────────────────────────
    team_mask = df["team_name"].astype(str) == str(team_name)

    mask = (
        team_mask &
        et.isin(["pass", "carry", "dribble"]) &
        df["x"].notna() & df["y"].notna() &
        df["end_x"].notna() & df["end_y"].notna()
    )
    actions = df[mask].copy()

    if actions.empty:
        return pd.DataFrame(columns=[
            "player_id","player_name","event_type",
            "x","y","end_x","end_y","xT_start","xT_end","xT_added",
        ])

    def _xt(x_val, y_val):
        try:
            c, r = coords_to_grid(float(x_val), float(y_val))
            return float(xT_grid[c, r])
        except Exception:
            return 0.0

    actions["xT_start"] = actions.apply(lambda r: _xt(r["x"],     r["y"]),     axis=1)
    actions["xT_end"]   = actions.apply(lambda r: _xt(r["end_x"], r["end_y"]), axis=1)
    actions["xT_added"] = actions["xT_end"] - actions["xT_start"]

    out_cols = [c for c in [
        "player_id","player_name","event_type",
        "x","y","end_x","end_y","xT_start","xT_end","xT_added",
    ] if c in actions.columns]
    return actions[out_cols].reset_index(drop=True)


# ============================================================================
# VISUALISATIONS (matplotlib / mplsoccer)
# ============================================================================

def plot_xt_grid(xT_grid: np.ndarray, title: str = "Expected Threat (xT) Grid") -> plt.Figure:
    fig, ax, _ = _make_pitch()

    cmap = LinearSegmentedColormap.from_list("xt_cyber", [
        PITCH_BG, "#1f2833", "#0d4f5c", CYAN_NEON, WHITE_GLOW,
    ])
    vmax = float(xT_grid.max()) if xT_grid.max() > 0 else 0.1

    im = ax.imshow(
        xT_grid.T, origin="lower",
        extent=[0, PITCH_W, 0, PITCH_H],
        aspect="auto", cmap=cmap, alpha=0.85, zorder=2,
        vmin=0, vmax=vmax,
    )

    for c in range(XT_COLS):
        for r in range(XT_ROWS):
            val = float(xT_grid[c, r])
            if val < 0.005: continue
            cx, cy = grid_to_center(c, r)
            col    = WHITE_GLOW if val > vmax * 0.5 else CYAN_NEON
            ax.text(cx, cy, f"{val:.3f}", color=col, fontsize=5.5,
                    ha="center", va="center", zorder=10,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground=PITCH_BG)])

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(colors=CYAN_NEON, labelsize=7)
    cbar.set_label("xT Value", color=CYAN_NEON, fontsize=9)
    cbar.ax.yaxis.label.set_color(CYAN_NEON)

    fig.text(0.5, 0.97, title, color=CYAN_NEON, fontsize=14, fontweight="bold",
             ha="center", va="top",
             path_effects=[pe.withStroke(linewidth=3, foreground=PITCH_BG)])
    plt.tight_layout()
    return fig


def plot_threat_added(threat_df: pd.DataFrame, team_name: str,
                      top_n: int = 20) -> plt.Figure:
    fig, ax, _ = _make_pitch()

    positive = threat_df[threat_df["xT_added"] > 0].nlargest(top_n, "xT_added") \
                         if not threat_df.empty else pd.DataFrame()

    if positive.empty:
        ax.text(60, 40, "No positive threat-added actions in this window.",
                color=CYAN_NEON, ha="center", va="center", fontsize=11,
                path_effects=[pe.withStroke(linewidth=2, foreground=PITCH_BG)])
        fig.text(0.5, 0.97, f"Top {top_n} Threat-Added — {team_name}",
                 color=CYAN_NEON, fontsize=13, fontweight="bold", ha="center", va="top",
                 path_effects=[pe.withStroke(linewidth=3, foreground=PITCH_BG)])
        plt.tight_layout()
        return fig

    norm = Normalize(vmin=0, vmax=float(positive["xT_added"].max()))
    for _, action in positive.iterrows():
        intensity  = float(norm(action["xT_added"]))
        alpha_base = 0.25 + 0.75 * intensity
        lw_base    = 0.6  + 3.0  * intensity

        for layer in (4, 3, 2, 1):
            ax.annotate("",
                xy=(float(action["end_x"]), float(action["end_y"])),
                xytext=(float(action["x"]), float(action["y"])),
                arrowprops=dict(
                    arrowstyle="-|>", color=CYAN_NEON,
                    lw=lw_base * (1 + (layer - 1) * 1.4),
                    alpha=min(alpha_base / layer, 1.0),
                ),
                zorder=4 + (4 - layer),
            )

    ax.scatter(positive["x"].astype(float), positive["y"].astype(float),
               s=35, color=PINK_NEON, alpha=0.9, zorder=12, linewidths=0)

    fig.text(0.5, 0.97, f"Top {top_n} Threat-Added — {team_name}",
             color=CYAN_NEON, fontsize=13, fontweight="bold", ha="center", va="top",
             path_effects=[pe.withStroke(linewidth=3, foreground=PITCH_BG)])
    plt.tight_layout()
    return fig