# player_stats.py Player Stats Radar Chart FIFA 25 Dataset
# player comparisons with radar charts showing Pace, Shooting, Passing, Dribbling, Defending, Physicality

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

STATS_COLS = {
    "pac": "Pace",
    "sho": "Shooting",
    "pas": "Passing",
    "dri": "Dribbling",
    "def": "Defending",
    "phy": "Physicality"
}

PLUS_MINUS_COLS = {
    "pac": "pac+-",
    "sho": "sho+-",
    "pas": "pas+-",
    "dri": "dri+-",
    "def": "def+-",
    "phy": "phy+-"
}

THEME_COLORS = {
    "PITCH_BG": "#0b0c10",
    "CYAN": "#66fcf1",
    "PINK": "#ff00ff",
    "GOLD": "#f5c518",
    "WHITE": "#ffffff",
    "DIM": "#445566"
}


# data
@st.cache_data(show_spinner="Loading player data…")
def load_players_data(filepath: str) -> pd.DataFrame:
    """Load and cache the FIFA 25 players dataset."""
    df = pd.read_csv(filepath)
    # Clean up any whitespace in column names
    df.columns = df.columns.str.strip()
    return df


# sparklines
def create_sparkline(current_value: float, change_value: float) -> go.Figure:
    previous_value = current_value - change_value
    if change_value > 0:
        line_color = "#00ff88"
    elif change_value < 0:
        line_color = "#ff4444"
    else:
        line_color = THEME_COLORS["DIM"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[previous_value, current_value],
        mode="lines+markers",
        line=dict(color=line_color, width=2),
        marker=dict(size=5, color=line_color),
        fill="tozeroy",
        fillcolor=f"rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.15)",
        hovertemplate="Previous: %{customdata[0]:.0f}<br>Current: %{customdata[1]:.0f}<extra></extra>",
        customdata=[[previous_value, current_value]],
        showlegend=False,
    ))
    
    fig.update_layout(
        height=60,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        hovermode="x unified",
    )
    
    return fig



# RADAR CHART
def build_player_radar(
    player_data: pd.Series,
    player_name: str,
) -> go.Figure:
    # stat values
    stat_values = []
    stat_labels = []
    
    for stat_key, stat_label in STATS_COLS.items():
        val = player_data.get(stat_key, 0)
        if pd.isna(val):
            val = 0
        stat_values.append(float(val))
        stat_labels.append(stat_label)
    
    # Close the loop for radar chart
    stat_values += stat_values[:1]
    stat_labels_loop = stat_labels + [stat_labels[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=stat_values,
        theta=stat_labels_loop,
        fill="toself",
        fillcolor="rgba(102, 252, 241, 0.25)",  # Cyan with transparency
        line=dict(color=THEME_COLORS["CYAN"], width=2),
        marker=dict(size=8, color=THEME_COLORS["CYAN"]),
        name=player_name,
        hovertemplate="<b>%{theta}</b><br>Rating: %{r:.0f}<extra></extra>"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color=THEME_COLORS["DIM"]),
                gridcolor=THEME_COLORS["DIM"],
                linecolor=THEME_COLORS["DIM"],
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=THEME_COLORS["WHITE"]),
                linecolor=THEME_COLORS["DIM"],
            ),
            bgcolor=THEME_COLORS["PITCH_BG"],
        ),
        paper_bgcolor=THEME_COLORS["PITCH_BG"],
        plot_bgcolor=THEME_COLORS["PITCH_BG"],
        font=dict(color=THEME_COLORS["WHITE"], size=12, family="Courier New"),
        height=550,
        margin=dict(l=80, r=80, t=80, b=80),
        showlegend=False,
        title=dict(
            text=f"<b>{player_name}</b>",
            font=dict(size=18, color=THEME_COLORS["CYAN"], family="Courier New"),
            x=0.5,
            xanchor="center",
        ),
    )
    
    return fig


def build_comparison_radar(
    player_data_1: pd.Series,
    player_name_1: str,
    player_data_2: pd.Series,
    player_name_2: str,
) -> go.Figure:
    stat_labels = list(STATS_COLS.values())
    
    #values for both players
    values_1 = []
    values_2 = []
    for stat_key in STATS_COLS.keys():
        val1 = player_data_1.get(stat_key, 0)
        val2 = player_data_2.get(stat_key, 0)
        if pd.isna(val1): val1 = 0
        if pd.isna(val2): val2 = 0
        values_1.append(float(val1))
        values_2.append(float(val2))
    
    # Close the loop
    values_1 += values_1[:1]
    values_2 += values_2[:1]
    stat_labels_loop = stat_labels + [stat_labels[0]]
    
    fig = go.Figure()
    
    # Player 1
    fig.add_trace(go.Scatterpolar(
        r=values_1,
        theta=stat_labels_loop,
        fill="toself",
        fillcolor="rgba(102, 252, 241, 0.25)",  # Cyan
        line=dict(color=THEME_COLORS["CYAN"], width=2),
        marker=dict(size=8, color=THEME_COLORS["CYAN"]),
        name=player_name_1,
        hovertemplate="<b>%{theta}</b><br>" + player_name_1 + ": %{r:.0f}<extra></extra>"
    ))
    
    # Player 2
    fig.add_trace(go.Scatterpolar(
        r=values_2,
        theta=stat_labels_loop,
        fill="toself",
        fillcolor="rgba(255, 0, 255, 0.25)",  # Pink
        line=dict(color=THEME_COLORS["PINK"], width=2),
        marker=dict(size=8, color=THEME_COLORS["PINK"]),
        name=player_name_2,
        hovertemplate="<b>%{theta}</b><br>" + player_name_2 + ": %{r:.0f}<extra></extra>"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color=THEME_COLORS["DIM"]),
                gridcolor=THEME_COLORS["DIM"],
                linecolor=THEME_COLORS["DIM"],
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=THEME_COLORS["WHITE"]),
                linecolor=THEME_COLORS["DIM"],
            ),
            bgcolor=THEME_COLORS["PITCH_BG"],
        ),
        paper_bgcolor=THEME_COLORS["PITCH_BG"],
        plot_bgcolor=THEME_COLORS["PITCH_BG"],
        font=dict(color=THEME_COLORS["WHITE"], size=12, family="Courier New"),
        height=550,
        margin=dict(l=80, r=80, t=80, b=80),
        legend=dict(
            x=1.1,
            y=1,
            bgcolor="rgba(13, 17, 23, 0.8)",
            bordercolor=THEME_COLORS["CYAN"],
            borderwidth=1,
            font=dict(color=THEME_COLORS["WHITE"]),
        ),
        title=dict(
            text=f"<b>{player_name_1} vs {player_name_2}</b>",
            font=dict(size=18, color=THEME_COLORS["CYAN"], family="Courier New"),
            x=0.5,
            xanchor="center",
        ),
    )
    
    return fig



# PLAYER PAGE
def render_player_stats_page(csv_path: str):
    st.markdown("## Player Statistics — FIFA")
    st.markdown("---")
    
    # Load data
    df = load_players_data(csv_path)
    
    # Mode selection
    mode = st.radio(
        "Select mode",
        ["Single Player", "Compare Two Players"],
        horizontal=True,
        key="player_mode"
    )
    
    st.markdown("---")
    
    if mode == "Single Player":
        # Single player view
        player_list = df["player_name"].tolist()
        selected_player_name = st.selectbox(
            "Select Player",
            player_list,
            key="single_player_select"
        )
        
        # Get player data
        player_row = df[df["player_name"] == selected_player_name].iloc[0]
        
        # Display player info and image
        col_info, col_radar = st.columns([1, 2])
        
        with col_info:
            st.markdown(f"### {selected_player_name}")
            st.markdown(f"**Club:** {player_row.get('club', 'N/A')}")
            st.markdown(f"**Position:** {player_row.get('position', 'N/A')}")
            st.markdown(f"**Nationality:** {player_row.get('nationality', 'N/A')}")
            st.markdown(f"**Age:** {player_row.get('age', 'N/A')}")
            st.markdown(f"**Overall Rating:** {player_row.get('ovr', 'N/A')}")
            
            # Display player image if available
            image_url = player_row.get("image_url", "")
            if image_url and isinstance(image_url, str) and image_url.startswith("http"):
                st.image(image_url, caption=selected_player_name, width=200)
        
        with col_radar:
            fig = build_player_radar(player_row, selected_player_name)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        # Stats table with sparklines
        st.markdown("---")
        st.markdown("### Detailed Stats")
        st.markdown("*Sparkline shows trend from previous year (green = up, red = down)*")
        
        # Create stats display with sparklines
        stats_cols_display = st.columns([1, 1, 2])
        with stats_cols_display[0]:
            st.markdown("**Attribute**")
        with stats_cols_display[1]:
            st.markdown("**Rating**")
        with stats_cols_display[2]:
            st.markdown("**Trend (Sparkline)**")
        
        st.divider()
        
        for stat_key, stat_label in STATS_COLS.items():
            val = player_row.get(stat_key, 0)
            change_col = PLUS_MINUS_COLS[stat_key]
            change_val = player_row.get(change_col, 0)
            
            if pd.isna(val): val = 0
            if pd.isna(change_val): change_val = 0
            
            val = int(val)
            change_val = int(change_val)
            
            # Determine trend color
            if change_val > 0:
                trend_color = "#00ff88"
                trend_str = f"+{change_val}"
            elif change_val < 0:
                trend_color = "#ff4444"
                trend_str = str(change_val)
            else:
                trend_color = THEME_COLORS["DIM"]
                trend_str = "−"
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                st.markdown(stat_label)
            
            with col2:
                st.markdown(f"<span style='color:#66fcf1;font-size:1.2rem'>{val}</span>", unsafe_allow_html=True)
            
            with col3:
                # Create and display sparkline
                fig = create_sparkline(val, change_val)
                # Display sparkline with trend indicator
                col3a, col3b = st.columns([4, 1])
                with col3a:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                with col3b:
                    st.markdown(f"<span style='color:{trend_color};font-size:1.1rem;font-weight:bold'>{trend_str}</span>", unsafe_allow_html=True)
    
    else:
        # Comparison mode
        col1, col2 = st.columns(2)
        
        player_list = df["player_name"].tolist()
        
        with col1:
            player_1_name = st.selectbox(
                "Select First Player",
                player_list,
                key="player_1_select"
            )
        
        with col2:
            # Default to second player if available
            default_idx = 1 if len(player_list) > 1 else 0
            player_2_name = st.selectbox(
                "Select Second Player",
                player_list,
                index=default_idx,
                key="player_2_select"
            )
        
        if player_1_name != player_2_name:
            player_1_row = df[df["player_name"] == player_1_name].iloc[0]
            player_2_row = df[df["player_name"] == player_2_name].iloc[0]
            
            # Display comparison radar
            fig = build_comparison_radar(
                player_1_row, player_1_name,
                player_2_row, player_2_name
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("---")
            st.markdown("### Comparison Stats with Trends")
            st.markdown("*Year-over-year trends*")
            
            # Header row
            comp_cols = st.columns([1.2, 1.2, 1.5, 1.2, 1.5])
            with comp_cols[0]:
                st.markdown("**Attribute**")
            with comp_cols[1]:
                st.markdown(f"**{player_1_name}**")
            with comp_cols[2]:
                st.markdown(f"**{player_1_name} Trend**")
            with comp_cols[3]:
                st.markdown(f"**{player_2_name}**")
            with comp_cols[4]:
                st.markdown(f"**{player_2_name} Trend**")
            
            st.divider()
            
            for stat_key, stat_label in STATS_COLS.items():
                val1 = player_1_row.get(stat_key, 0)
                val2 = player_2_row.get(stat_key, 0)
                change_col = PLUS_MINUS_COLS[stat_key]
                change_val1 = player_1_row.get(change_col, 0)
                change_val2 = player_2_row.get(change_col, 0)
                
                if pd.isna(val1): val1 = 0
                if pd.isna(val2): val2 = 0
                if pd.isna(change_val1): change_val1 = 0
                if pd.isna(change_val2): change_val2 = 0
                
                val1 = int(val1)
                val2 = int(val2)
                change_val1 = int(change_val1)
                change_val2 = int(change_val2)
                
                # Determine trend colors
                trend_color_1 = "#00ff88" if change_val1 > 0 else ("#ff4444" if change_val1 < 0 else THEME_COLORS["DIM"])
                trend_color_2 = "#00ff88" if change_val2 > 0 else ("#ff4444" if change_val2 < 0 else THEME_COLORS["DIM"])
                trend_str_1 = f"+{change_val1}" if change_val1 > 0 else (str(change_val1) if change_val1 < 0 else "−")
                trend_str_2 = f"+{change_val2}" if change_val2 > 0 else (str(change_val2) if change_val2 < 0 else "−")
                
                comp_cols = st.columns([1.2, 1.2, 1.5, 1.2, 1.5])
                
                with comp_cols[0]:
                    st.markdown(stat_label)
                
                with comp_cols[1]:
                    st.markdown(f"<span style='color:#66fcf1;font-weight:bold'>{val1}</span>", unsafe_allow_html=True)
                
                with comp_cols[2]:
                    fig1 = create_sparkline(val1, change_val1)
                    col_spark, col_trend = st.columns([3, 1])
                    with col_spark:
                        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
                    with col_trend:
                        st.markdown(f"<span style='color:{trend_color_1};font-weight:bold'>{trend_str_1}</span>", unsafe_allow_html=True)
                
                with comp_cols[3]:
                    st.markdown(f"<span style='color:#ff00ff;font-weight:bold'>{val2}</span>", unsafe_allow_html=True)
                
                with comp_cols[4]:
                    fig2 = create_sparkline(val2, change_val2)
                    col_spark, col_trend = st.columns([3, 1])
                    with col_spark:
                        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
                    with col_trend:
                        st.markdown(f"<span style='color:{trend_color_2};font-weight:bold'>{trend_str_2}</span>", unsafe_allow_html=True)
        else:
            st.warning("Select two different players to compare.")


# STATSBOMB PLAYER STATS

from collections import Counter, defaultdict
import requests

from data_engine import get_match_events, get_match_list
from xT_model import build_transition_matrix, solve_xt


SB_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
SB_COMPETITION_NAME = "UEFA Euro 2024"
SB_COMPETITION_ID = 55
SB_SEASON_ID = 282

SB_THEME = {
    "bg": "#ffffff",
    "text": "#6b6b6b",
    "muted": "#9a9a9a",
    "red": "#66fcf1",
    "gold": "#ff00ff",
    "light": "#eef2f7",
    "stripe": "#f7f9fc",
    "grid": "#d7dce5",
    "dark": "#444444",
}

RANK_METRICS = [
    ("xg_per90", "xG"),
    ("shots_per90", "Shots"),
    ("shot_xg_per_shot", "xG/Shot"),
    ("pressures_per90", "Pressures"),
    ("pass_xt_per90", "Pass xT"),
    ("open_play_xg_assisted_per90", "Open Play xG Assisted"),
    ("carry_dribble_xt_per90", "Dribble & Carry xT"),
    ("progressive_passes_per90", "Progressive Passes"),
    ("fouls_won_per90", "Fouls Won"),
    ("turnovers_per90", "Turnovers"),
    ("touches_in_box_per90", "Touches in Box"),
]

RADAR_METRICS = [
    ("xg_per90_pct", "xG"),
    ("shots_per90_pct", "Shots"),
    ("pressures_per90_pct", "Pressures"),
    ("pass_xt_per90_pct", "Pass xT"),
    ("carry_dribble_xt_per90_pct", "Dribble & Carry xT"),
    ("touches_in_box_per90_pct", "Touches in Box"),
]


def _fetch_json(url: str):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def _period_start_seconds(period: int) -> int:
    offsets = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60}
    return offsets.get(int(period or 1), 0)


def _clock_to_seconds(clock: str | None) -> float:
    if not clock:
        return np.nan
    try:
        minute_str, second_str = str(clock).split(":")[:2]
        return float(minute_str) * 60.0 + float(second_str)
    except Exception:
        return np.nan


def _absolute_seconds(clock: str | None, period: int | None, default: float | None = None) -> float:
    value = _clock_to_seconds(clock)
    if np.isnan(value):
        return float(default) if default is not None else np.nan
    return value + _period_start_seconds(period)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator and not np.isnan(denominator):
        return float(numerator) / float(denominator)
    return 0.0


def _is_box_touch(x_value, y_value) -> bool:
    if pd.isna(x_value) or pd.isna(y_value):
        return False
    return float(x_value) >= 102.0 and 18.0 <= float(y_value) <= 62.0


def _is_progressive_pass(row: pd.Series) -> bool:
    if row.get("event_type") != "Pass":
        return False
    if str(row.get("result", "")).lower() != "complete":
        return False
    if pd.isna(row.get("x")) or pd.isna(row.get("end_x")):
        return False
    try:
        start_x = float(row["x"])
        end_x = float(row["end_x"])
    except Exception:
        return False
    return (end_x - start_x) >= 15.0 or end_x >= 80.0


@st.cache_data(show_spinner="Loading StatsBomb player summaries…")
def load_statsbomb_player_summary() -> pd.DataFrame:
    matches_df = get_match_list()
    if matches_df.empty:
        return pd.DataFrame()

    event_frames = []
    player_meta = {}
    player_minutes = defaultdict(float)
    player_appearances = defaultdict(set)
    position_counts = defaultdict(Counter)
    team_counts = defaultdict(Counter)
    jersey_counts = defaultdict(Counter)
    country_counts = defaultdict(Counter)

    for match in matches_df.itertuples(index=False):
        match_id = int(match.match_id)
        events_df = get_match_events(match_id).copy()
        if not events_df.empty:
            events_df["match_id"] = match_id
            event_frames.append(events_df)

        try:
            raw_lineups = _fetch_json(f"{SB_BASE_URL}/lineups/{match_id}.json")
        except Exception:
            raw_lineups = []

        match_end_seconds = 90.0 * 60.0
        if not events_df.empty and {"minute", "second"}.issubset(events_df.columns):
            match_end_seconds = float(
                (events_df["minute"].fillna(0) * 60.0 + events_df["second"].fillna(0)).max()
            )

        for team in raw_lineups:
            team_name = team.get("team_name", "")
            for player in team.get("lineup", []):
                player_id = str(player.get("player_id", ""))
                player_name = player.get("player_name", "")
                if not player_id or not player_name:
                    continue

                player_meta.setdefault(player_id, {})
                player_meta[player_id]["player_id"] = player_id
                player_meta[player_id]["player_name"] = player_name
                player_meta[player_id]["team_name"] = team_name
                player_meta[player_id]["jersey_number"] = player.get("jersey_number", "")
                player_meta[player_id]["country"] = (player.get("country") or {}).get("name", "")

                team_counts[player_id][team_name] += 1
                jersey_counts[player_id][str(player.get("jersey_number", ""))] += 1
                country_counts[player_id][(player.get("country") or {}).get("name", "")] += 1

                positions = player.get("positions", []) or []
                if positions:
                    player_appearances[player_id].add(match_id)

                for position in positions:
                    position_name = position.get("position", "Unknown")
                    position_counts[player_id][position_name] += 1
                    start_seconds = _absolute_seconds(
                        position.get("from"),
                        position.get("from_period"),
                    )
                    end_seconds = _absolute_seconds(
                        position.get("to"),
                        position.get("to_period"),
                        default=match_end_seconds,
                    )
                    if np.isnan(start_seconds):
                        continue
                    if np.isnan(end_seconds):
                        end_seconds = match_end_seconds
                    player_minutes[player_id] += max(0.0, end_seconds - start_seconds) / 60.0

    if not event_frames:
        return pd.DataFrame()

    all_events = pd.concat(event_frames, ignore_index=True)
    all_events = all_events.sort_values(["match_id", "index"]).reset_index(drop=True)

    T, p_shoot, p_goal = build_transition_matrix(all_events)
    xT_grid = solve_xt(T, p_shoot, p_goal)

    def _xt_at(x_value, y_value) -> float:
        if pd.isna(x_value) or pd.isna(y_value):
            return 0.0
        try:
            col = int(np.clip(int(float(x_value) / 120.0 * 12), 0, 11))
            row = int(np.clip(int(float(y_value) / 80.0 * 8), 0, 7))
            return float(xT_grid[col, row])
        except Exception:
            return 0.0

    player_totals = defaultdict(lambda: defaultdict(float))

    for _, row in all_events.iterrows():
        player_id = str(row.get("player_id", ""))
        if not player_id:
            continue

        player_name = row.get("player_name", "")
        team_name = row.get("team_name", "")
        player_totals[player_id]["player_name"] = player_name
        player_totals[player_id]["team_name"] = team_name
        player_totals[player_id]["shirt_number"] = player_meta.get(player_id, {}).get("jersey_number", "")

        event_type = str(row.get("event_type", ""))
        result = str(row.get("result", ""))
        x_value = row.get("x")
        y_value = row.get("y")
        end_x_value = row.get("end_x")
        end_y_value = row.get("end_y")

        if event_type == "Shot":
            player_totals[player_id]["shots"] += 1
            player_totals[player_id]["xg"] += float(row.get("statsbomb_xg", 0.0) or 0.0)

        if event_type == "Pressure":
            player_totals[player_id]["pressures"] += 1

        if event_type == "Foul Won":
            player_totals[player_id]["fouls_won"] += 1

        if _is_box_touch(x_value, y_value):
            player_totals[player_id]["touches_in_box"] += 1

        if event_type == "Pass":
            if result.lower() != "complete":
                player_totals[player_id]["turnovers"] += 1
            else:
                if _is_progressive_pass(row):
                    player_totals[player_id]["progressive_passes"] += 1

        if event_type in {"Carry", "Dribble"} and result.lower() != "complete":
            player_totals[player_id]["turnovers"] += 1

        if event_type in {"Dispossessed", "Miscontrol"}:
            player_totals[player_id]["turnovers"] += 1

        if event_type == "Pass" and result.lower() == "complete" and not any(
            pd.isna(value) for value in [x_value, y_value, end_x_value, end_y_value]
        ):
            player_totals[player_id]["pass_xt"] += _xt_at(end_x_value, end_y_value) - _xt_at(x_value, y_value)

        if event_type in {"Carry", "Dribble"} and not any(
            pd.isna(value) for value in [x_value, y_value, end_x_value, end_y_value]
        ):
            player_totals[player_id]["carry_dribble_xt"] += _xt_at(end_x_value, end_y_value) - _xt_at(x_value, y_value)

    for match_id, match_df in all_events.groupby("match_id", sort=False):
        match_df = match_df.sort_values("index").reset_index(drop=True)
        for idx, shot_row in match_df.iterrows():
            if shot_row.get("event_type") != "Shot":
                continue
            if str(shot_row.get("play_pattern", "")).lower() != "open play":
                continue

            shot_xg = float(shot_row.get("statsbomb_xg", 0.0) or 0.0)
            if shot_xg <= 0:
                continue

            same_possession = match_df.iloc[:idx]
            same_possession = same_possession[
                (same_possession["possession"] == shot_row.get("possession"))
                & (same_possession["team_name"] == shot_row.get("team_name"))
            ]
            if same_possession.empty:
                continue

            pass_chain = same_possession[same_possession["event_type"] == "Pass"]
            if not pass_chain.empty:
                assister = pass_chain.iloc[-1]
                assister_id = str(assister.get("player_id", ""))
                if assister_id:
                    player_totals[assister_id]["open_play_xg_assisted"] += shot_xg

            carry_chain = same_possession[same_possession["event_type"].isin(["Carry", "Dribble"])]
            if not carry_chain.empty:
                carrier = carry_chain.iloc[-1]
                carrier_id = str(carrier.get("player_id", ""))
                if carrier_id:
                    player_totals[carrier_id]["carry_dribble_xg"] += shot_xg

    rows = []
    for player_id, totals in player_totals.items():
        minutes_played = float(player_minutes.get(player_id, 0.0))
        nineties_played = minutes_played / 90.0 if minutes_played > 0 else 0.0
        if minutes_played <= 0:
            continue

        shots_total = float(totals.get("shots", 0.0))
        xg_total = float(totals.get("xg", 0.0))

        row = {
            "player_id": player_id,
            "player_name": totals.get("player_name", player_meta.get(player_id, {}).get("player_name", "")),
            "team_name": totals.get("team_name", player_meta.get(player_id, {}).get("team_name", "")),
            "shirt_number": totals.get("shirt_number", player_meta.get(player_id, {}).get("jersey_number", "")),
            "country": player_meta.get(player_id, {}).get("country", ""),
            "position": position_counts[player_id].most_common(1)[0][0] if position_counts[player_id] else "",
            "appearances": len(player_appearances[player_id]) if player_appearances[player_id] else 0,
            "minutes_played": minutes_played,
            "nineties_played": nineties_played,
            "xg_per90": _safe_div(xg_total, nineties_played),
            "shots_per90": _safe_div(shots_total, nineties_played),
            "shot_xg_per_shot": _safe_div(xg_total, shots_total),
            "pressures_per90": _safe_div(totals.get("pressures", 0.0), nineties_played),
            "pass_xt_per90": _safe_div(totals.get("pass_xt", 0.0), nineties_played),
            "open_play_xg_assisted_per90": _safe_div(totals.get("open_play_xg_assisted", 0.0), nineties_played),
            "carry_dribble_xt_per90": _safe_div(totals.get("carry_dribble_xt", 0.0), nineties_played),
            "progressive_passes_per90": _safe_div(totals.get("progressive_passes", 0.0), nineties_played),
            "fouls_won_per90": _safe_div(totals.get("fouls_won", 0.0), nineties_played),
            "turnovers_per90": _safe_div(totals.get("turnovers", 0.0), nineties_played),
            "touches_in_box_per90": _safe_div(totals.get("touches_in_box", 0.0), nineties_played),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    percentile_base = summary_df[summary_df["minutes_played"] >= 90].copy()
    if len(percentile_base) < 15:
        percentile_base = summary_df.copy()

    for metric_key, _ in RANK_METRICS:
        percentile_col = f"{metric_key}_pct"
        summary_df[percentile_col] = summary_df[metric_key].rank(pct=True, method="average") * 100.0
        if not percentile_base.empty:
            lookup = (
                percentile_base[["player_id", metric_key]]
                .assign(_pct=lambda frame: frame[metric_key].rank(pct=True, method="average") * 100.0)
                .set_index("player_id")
                ["_pct"]
                .to_dict()
            )
            summary_df[percentile_col] = summary_df["player_id"].map(lookup).fillna(summary_df[percentile_col])

    summary_df = summary_df.sort_values(["team_name", "minutes_played", "player_name"], ascending=[True, False, True]).reset_index(drop=True)
    return summary_df


def _format_value(metric_key: str, value: float) -> str:
    if metric_key == "shot_xg_per_shot":
        return f"{value:.2f}"
    if metric_key in {"xg_per90", "pass_xt_per90", "open_play_xg_assisted_per90", "carry_dribble_xt_per90"}:
        return f"{value:.2f}"
    return f"{value:.2f}"


def _build_radar_figure(player_row: pd.Series) -> go.Figure:
    theta = [label for _, label in RADAR_METRICS]
    values = [float(player_row.get(metric_key, 0.0) or 0.0) for metric_key, _ in RADAR_METRICS]
    values += values[:1]
    theta_loop = theta + [theta[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=theta_loop,
            fill="toself",
            fillcolor="rgba(192, 57, 43, 0.30)",
            line=dict(color=SB_THEME["red"], width=3),
            marker=dict(size=8, color=SB_THEME["gold"]),
            hovertemplate="<b>%{theta}</b><br>Percentile: %{r:.0f}<extra></extra>",
            name=player_row.get("player_name", "Player"),
            showlegend=False,
        )
    )

    fig.update_layout(
        height=560,
        margin=dict(l=50, r=50, t=40, b=40),
        paper_bgcolor=SB_THEME["bg"],
        plot_bgcolor=SB_THEME["bg"],
        font=dict(color=SB_THEME["dark"], family="Arial"),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color=SB_THEME["muted"], size=10),
                gridcolor=SB_THEME["grid"],
                linecolor=SB_THEME["grid"],
            ),
            angularaxis=dict(
                tickfont=dict(color=SB_THEME["dark"], size=11),
                linecolor=SB_THEME["grid"],
            ),
            bgcolor=SB_THEME["bg"],
        ),
    )
    return fig


def _build_metrics_table(player_row: pd.Series, player_name: str) -> str:
    rows = []
    for metric_key, metric_label in RANK_METRICS:
        value = float(player_row.get(metric_key, 0.0) or 0.0)
        percentile = float(player_row.get(f"{metric_key}_pct", 0.0) or 0.0)
        rows.append(
            f"<tr>"
            f"<td style='padding:8px 12px;color:{SB_THEME['text']};font-size:0.98rem'>{metric_label}</td>"
            f"<td style='padding:8px 12px;color:{SB_THEME['dark']};font-weight:600;text-align:right'>{_format_value(metric_key, value)}</td>"
            f"<td style='padding:8px 12px;color:{SB_THEME['red'] if percentile >= 50 else SB_THEME['gold']};font-weight:700;text-align:right'>{percentile:.0f}</td>"
            f"</tr>"
        )

    return (
        "<table style='width:100%;border-collapse:collapse;border:1px solid #d7dce5;border-radius:10px;overflow:hidden;font-family:Arial,sans-serif;background:#ffffff'>"
        "<thead>"
        f"<tr style='background:#a8a8a8;color:#ffffff'>"
        f"<th style='padding:10px 12px;text-align:left;font-size:0.95rem'>{player_name}</th>"
        f"<th style='padding:10px 12px;text-align:right;font-size:0.95rem'>Value</th>"
        f"<th style='padding:10px 12px;text-align:right;font-size:0.95rem'>Percentile</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "<div style='text-align:right;color:#b0b0b0;font-style:italic;font-size:0.85rem;margin-top:6px'>All units in per-90 unless noted</div>"
    )


def render_player_stats_page():
    st.markdown(
        "<div style='display:flex;justify-content:space-between;align-items:flex-end;gap:16px'>"
        f"<div><div style='font-size:2rem;color:{SB_THEME['red']};font-weight:800;line-height:1.05'>StatsBomb Player Radar</div>"
        f"<div style='font-size:1.05rem;color:{SB_THEME['gold']};font-style:italic'>UEFA Euro 2024</div></div>"
        f"<div style='text-align:right;color:{SB_THEME['muted']};font-size:1rem;font-style:italic'>Tournament-wide player summary</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    with st.spinner("Loading StatsBomb tournament player summaries…"):
        summary_df = load_statsbomb_player_summary()

    if summary_df.empty:
        st.info("No StatsBomb player data was available for UEFA Euro 2024.")
        return

    team_options = ["All Teams"] + sorted(summary_df["team_name"].dropna().unique().tolist())
    team_choice = st.selectbox("Filter team", team_options, index=0)

    filtered_df = summary_df.copy()
    if team_choice != "All Teams":
        filtered_df = filtered_df[filtered_df["team_name"] == team_choice].copy()

    filtered_df = filtered_df.sort_values(["team_name", "minutes_played", "player_name"], ascending=[True, False, True]).reset_index(drop=True)
    player_options = filtered_df["player_name"].tolist()
    if not player_options:
        st.warning("No players found for that team filter.")
        return

    default_index = 0
    selected_player_name = st.selectbox("Select player", player_options, index=default_index)
    player_row = filtered_df[filtered_df["player_name"] == selected_player_name].iloc[0]

    left_col, right_col = st.columns([1.0, 1.15], vertical_alignment="top")

    with left_col:
        st.markdown(
            f"<div style='color:{SB_THEME['red']};font-size:2.1rem;font-weight:800;line-height:1.0'>{selected_player_name}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:{SB_THEME['gold']};font-size:1.15rem;font-style:italic'>{player_row.get('team_name', 'N/A')}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:{SB_THEME['gold']};font-size:1rem;font-style:italic'>UEFA Euro 2024</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:{SB_THEME['text']};font-size:0.95rem'>Age: N/A &nbsp;&nbsp;|&nbsp;&nbsp; Birth date: N/A</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(_build_radar_figure(player_row), use_container_width=True, config={"displayModeBar": False})

    with right_col:
        st.markdown(
            f"<div style='text-align:right;color:{SB_THEME['muted']};font-size:1.9rem;font-weight:700;font-style:italic;line-height:1.05'>StatsBomb Euro 2024 Player Radar and Information</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='text-align:right;color:{SB_THEME['text']};font-size:1.05rem'>"
            f"UEFA Euro 2024<br>{player_row.get('nineties_played', 0.0):.1f} 90s played ({int(player_row.get('appearances', 0))} appearances)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='border:1px solid #d7dce5;border-radius:10px;overflow:hidden;background:#ffffff'>"
            f"<table style='width:100%;border-collapse:collapse;font-family:Arial,sans-serif'>"
            f"<tr style='background:#9f9f9f;color:#ffffff'><th colspan='2' style='padding:9px 12px;text-align:center;font-size:1rem'>{selected_player_name}</th></tr>"
            f"<tr style='background:#f7f9fc'><td style='padding:8px 12px;color:{SB_THEME['muted']}'>90s Played</td><td style='padding:8px 12px;color:{SB_THEME['dark']};font-weight:600'>"
            f"{player_row.get('nineties_played', 0.0):.1f}</td></tr>"
            f"<tr><td style='padding:8px 12px;color:{SB_THEME['muted']}'>Competition</td><td style='padding:8px 12px;color:{SB_THEME['dark']}'>UEFA Euro</td></tr>"
            f"<tr style='background:#f7f9fc'><td style='padding:8px 12px;color:{SB_THEME['muted']}'>Season</td><td style='padding:8px 12px;color:{SB_THEME['dark']}'>2024</td></tr>"
            f"<tr><td style='padding:8px 12px;color:{SB_THEME['muted']}'>Match Type</td><td style='padding:8px 12px;color:{SB_THEME['dark']}'>All Matches</td></tr>"
            f"<tr style='background:#f7f9fc'><td style='padding:8px 12px;color:{SB_THEME['muted']}'>Team</td><td style='padding:8px 12px;color:{SB_THEME['dark']}'>{player_row.get('team_name', 'N/A')}</td></tr>"
            f"<tr><td style='padding:8px 12px;color:{SB_THEME['muted']}'>Position</td><td style='padding:8px 12px;color:{SB_THEME['dark']}'>{player_row.get('position', 'N/A')}</td></tr>"
            f"<tr style='background:#f7f9fc'><td style='padding:8px 12px;color:{SB_THEME['muted']}'>Shirt Number</td><td style='padding:8px 12px;color:{SB_THEME['dark']}'>{player_row.get('shirt_number', 'N/A')}</td></tr>"
            f"<tr><td style='padding:8px 12px;color:{SB_THEME['muted']}'>Appearances</td><td style='padding:8px 12px;color:{SB_THEME['dark']}'>{int(player_row.get('appearances', 0))}</td></tr>"
            f"</table></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(_build_metrics_table(player_row, selected_player_name), unsafe_allow_html=True)
