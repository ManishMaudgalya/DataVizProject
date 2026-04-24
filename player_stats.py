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
