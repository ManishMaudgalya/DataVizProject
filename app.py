# Main streamlit app.py
# Euro 2024 match analysis dashboard
# Player Stats FIFA 25 player radar comparison 


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

from data_engine   import (get_match_list, load_match_data,
                            get_teams, filter_by_time)
from pass_network  import (build_pass_network, plot_pass_network,
                            _base_pitch_figure, PITCH_BG, CYAN, PINK, GOLD, WHITE, DIM)
from pitch_control import (plot_pitch_control, plot_team_shape_single,
                            plot_team_shape_both)
from xT_model      import (build_transition_matrix, solve_xt,
                            compute_threat_added, plot_xt_grid, plot_threat_added)
from player_stats  import render_player_stats_page

#Page
st.set_page_config(
    page_title="Threat, Space & Control — Euro 2024",
    page_icon="⚽", layout="wide",
    initial_sidebar_state="expanded",
)

#Cyberpunk CSS
st.markdown("""
<style>
.stApp{background:#0b0c10}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #1f2833}
.stMarkdown,.stText,label,.stSelectbox label{color:#c5c6c7!important}
p,li{color:#c5c6c7}
h1,h2,h3{color:#66fcf1!important;font-family:'Courier New',monospace!important}
[data-testid="stMetricValue"]{color:#66fcf1!important;font-size:1.9rem!important}
[data-testid="stMetricLabel"]{color:#c5c6c7!important}
[data-testid="stMetricDelta"]{color:#ff00ff!important}
.stTabs [data-baseweb="tab"]{color:#c5c6c7;background:#1f2833;border-radius:6px 6px 0 0}
.stTabs [aria-selected="true"]{color:#66fcf1!important;border-bottom:2px solid #66fcf1!important}
.stSelectbox>div>div{background:#1f2833;color:#c5c6c7}
.stButton>button{background:#1f2833;color:#66fcf1;border:1px solid #66fcf1;transition:box-shadow .25s}
.stButton>button:hover{box-shadow:0 0 14px #66fcf1}
hr{border-color:#1f2833}
.stCaption,small{color:#445566!important}
/* prevent metric value truncation */
[data-testid="stMetricValue"]{overflow:visible!important;white-space:normal!important}
</style>
""", unsafe_allow_html=True)

#Config
_PLOT_CFG = {"displayModeBar": False, "responsive": True}

#Session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Euro 2024 Analytics"

# page selector on the sidebar
with st.sidebar:
    st.markdown("## ⚽ Analytics Dashboard")
    st.markdown("*Threat · Space · Control · Players*")
    st.markdown("---")
    
    page = st.radio(
        "Select Page",
        ["Euro 2024 Analytics", "Player Stats"],
        horizontal=False,
    )
    st.session_state.current_page = page
    st.markdown("---")


# Page route
if st.session_state.current_page == "Euro 2024 Analytics":
    # EURO 2024 ANALYTICS PAGE
    
    with st.sidebar:
        with st.spinner("Loading fixtures…"):
            matches_df = get_match_list()

        sel_label = st.selectbox("Select Match", matches_df["label"].tolist(), index=0)
        sel       = matches_df[matches_df["label"] == sel_label].iloc[0]
        match_id  = int(sel["match_id"])

        st.markdown(f"**Stage:** {sel['stage']}")
        st.markdown(f"**Date:**  {sel['match_date']}")
        st.markdown("---")

        with st.spinner("Streaming match data…"):
            events_df, freeze_frames = load_match_data(match_id)

        teams        = get_teams(events_df)
        sel_team     = st.selectbox("Analyze Team", teams, index=0)
        opp_team     = next((t for t in teams if t != sel_team), "Opponent")

        st.markdown("---")
        st.markdown("### ⏱ Time Window")

        half_sel = st.radio("Quick select",
            ["Full Match", "1st Half", "2nd Half", "Extra Time"],
            horizontal=True)

        max_min = int(events_df["minute"].max()) if not events_df.empty else 90
        default = {"Full Match": (0, max_min), "1st Half": (0, 45),
                   "2nd Half": (45, 90), "Extra Time": (90, max_min)}.get(half_sel, (0, max_min))

        minute_range = st.slider("Minute range", 0, max(max_min, 90),
                                 default, step=1,
                                 help="All visualisations update to this window")
        st.markdown("---")
        st.markdown("**Dataset:** StatsBomb Open Data")
        st.markdown("**Coords:** 120×80 yards (StatsBomb)")
        st.caption("Master's Capstone Project")

    # Filtered data
    fdf        = filter_by_time(events_df, minute_range[0], minute_range[1])
    min_label  = f"({minute_range[0]}–{minute_range[1]} min)"

    # Header
    st.markdown(f"# 🎯 {sel['label']}")
    st.markdown(f"*Analyzing **{sel_team}** · {min_label}*")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Events",       f"{len(fdf):,}")
    c2.metric("Passes",       f"{int((fdf['event_type']=='Pass').sum()):,}")
    c3.metric("Shots",        f"{int((fdf['event_type']=='Shot').sum()):,}")
    c4.metric("Carries",      f"{int((fdf['event_type']=='Carry').sum()):,}")
    c5.metric("360° Frames",  f"{len(freeze_frames):,}")
    st.markdown("---")


    # EVENT TIMELINE
    def _build_timeline(df: pd.DataFrame, minute_range: tuple) -> go.Figure:
        show_types = ["Pass","Shot","Carry","Pressure","Foul Committed",
                      "Ball Recovery","Duel"]
        colors = {"Pass":"#66fcf1","Shot":"#ff4444","Carry":"#445566",
                  "Pressure":"#9400d3","Foul Committed":"#ff8c00",
                  "Ball Recovery":"#00ff88","Duel":"#ffdd44"}

        fig = go.Figure()
        for et in show_types:
            sub = df[df["event_type"] == et]
            if sub.empty: continue
            t = sub["minute"] + sub["second"].fillna(0) / 60
            cd = np.stack([sub["minute"].values,
                           sub["second"].fillna(0).values.astype(int),
                           sub["team_name"].fillna("").values,
                           sub["player_name"].fillna("").values], axis=-1)
            fig.add_trace(go.Scatter(
                x=t, y=[et]*len(sub), mode="markers",
                marker=dict(size=5, color=colors.get(et,"#888"), opacity=0.75),
                name=et, customdata=cd,
                hovertemplate="<b>%{customdata[3]}</b> (%{customdata[2]})<br>"
                              "%{customdata[0]}:%{customdata[1]:02d}<extra></extra>",
            ))

        fig.add_vline(x=45, line_dash="dash", line_color="#2a3a4a", line_width=1.5)
        fig.add_annotation(x=45, y=len(show_types)-0.4, text="HT",
                           font=dict(color="#445566", size=9), showarrow=False)
        fig.add_vrect(x0=minute_range[0], x1=minute_range[1],
                      fillcolor="#66fcf1", opacity=0.04,
                      line=dict(color="#66fcf1", width=0.7))

        fig.update_layout(
            paper_bgcolor=PITCH_BG, plot_bgcolor=PITCH_BG, height=175,
            margin=dict(l=5, r=20, t=4, b=35),
            showlegend=False,
            xaxis=dict(
                title=dict(text="Match minute", font=dict(color="#445566", size=10)),
                range=[0, max(max_min, 93)],
                showgrid=False, color="#445566",
                tickfont=dict(color="#445566", size=9),
            ),
            yaxis=dict(showgrid=False, color="#445566",
                       tickfont=dict(color="#445566", size=9)),
            hoverlabel=dict(bgcolor="#1a2332", bordercolor=CYAN,
                            font=dict(color=WHITE, size=11)),
        )
        return fig

    st.plotly_chart(_build_timeline(events_df, minute_range),
                    use_container_width=True, config=_PLOT_CFG)
    st.markdown("---")


    
    # TABS
    tab_pass, tab_shape, tab_voronoi, tab_xt, tab_shots, tab_raw = st.tabs([
        "📡 Pass Network", "🧭 Team Shape", "🗺️ Pitch Control",
        "⚡ Expected Threat", "🎯 Shot Map", "🔍 Events Explorer",
    ])


    #PASS NETWORK
    with tab_pass:
        st.subheader(f"Pass Network — {sel_team}  {min_label}")
        st.markdown(
            "*Node = player's average passing position · "
            "Edge thickness = pass frequency · Hover nodes for partner stats*"
        )

        ctrl, viz = st.columns([1, 4])
        with ctrl:
            min_thresh = st.slider("Min passes to show edge", 1, 15, 2, key="pn_min")

        with viz:
            with st.spinner("Building pass network…"):
                try:
                    nodes_df, edges_df, best_partners = build_pass_network(
                        fdf, sel_team,
                        minute_min=minute_range[0], minute_max=minute_range[1],
                        min_passes=min_thresh,
                    )
                    fig_pass = plot_pass_network(
                        nodes_df, edges_df, best_partners,
                        team_name=sel_team, match_label=sel["label"],
                        minute_label=min_label,
                    )
                    st.plotly_chart(fig_pass, use_container_width=True, config=_PLOT_CFG)

                    # Summary — use markdown for long text so nothing gets truncated
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Players shown",    len(nodes_df) if not nodes_df.empty else 0)
                    m2.metric("Pass connections", len(edges_df) if not edges_df.empty else 0)
                    with m3:
                        if not edges_df.empty:
                            top = edges_df.loc[edges_df["pass_count"].idxmax()]
                            st.markdown("**Most linked pair**")
                            # Full names — never clipped because it's raw markdown
                            st.markdown(
                                f"<span style='color:#66fcf1;font-size:1.1rem'>"
                                f"{top.player_a_name} ↔ {top.player_b_name} "
                                f"({top.pass_count})</span>",
                                unsafe_allow_html=True,
                            )

                except Exception as e:
                    st.error(f"Pass network error: {e}")
                    st.caption("Try lowering the min-passes threshold or widening the time window.")


    #TEAM SHAPE
    with tab_shape:
        st.subheader(f"Team Shape — {sel_team}  {min_label}")
        st.markdown(
            "*Average position of each player across all their actions. "
            "Hover nodes for event counts.*"
        )

        show_both = st.checkbox("Show both teams side-by-side", value=False)

        with st.spinner("Computing team shape…"):
            try:
                if show_both:
                    fig_both = plot_team_shape_both(
                        fdf, sel_team, opp_team,
                        minute_min=minute_range[0], minute_max=minute_range[1],
                    )
                    st.plotly_chart(fig_both, use_container_width=True, config=_PLOT_CFG)
                else:
                    fig_shape = plot_team_shape_single(
                        fdf, sel_team,
                        minute_min=minute_range[0], minute_max=minute_range[1],
                    )
                    st.plotly_chart(fig_shape, use_container_width=True, config=_PLOT_CFG)
            except Exception as e:
                st.error(f"Team shape error: {e}")


    #PITCH CONTROL
    with tab_voronoi:
        st.subheader("Pitch Control — Voronoi Tessellation")
        st.markdown(
            f"*Hover player dots to see info.  "
            f"■ Cyan = {sel_team}  ·  ■ Pink = {opp_team}*"
        )

        ctrl3, viz3 = st.columns([1, 4])

        with ctrl3:
            if not freeze_frames:
                st.info("No 360° freeze-frame data for this match. "
                        "Use **Team Shape** tab for spatial analysis.")
                sel_ff   = []
                ev_desc  = ""
            else:
                ff_eids = list(freeze_frames.keys())
                ff_evts = events_df[events_df["event_id"].isin(ff_eids)].copy()

                if not ff_evts.empty:
                    ff_evts["opt"] = (
                        ff_evts["minute"].astype(str) + "' " +
                        ff_evts["event_type"] + " — " +
                        ff_evts["player_name"].fillna("Unknown")
                    )
                    choice  = st.selectbox("Select event moment", ff_evts["opt"].tolist())
                    sel_row = ff_evts[ff_evts["opt"] == choice].iloc[0]
                    sel_ff  = freeze_frames.get(sel_row["event_id"], [])
                    ev_desc = choice
                else:
                    chosen_key = st.selectbox("Frame ID", ff_eids[:40])
                    sel_ff     = freeze_frames[chosen_key]
                    ev_desc    = chosen_key[:30]

        with viz3:
            if freeze_frames:
                with st.spinner("Computing Voronoi…"):
                    try:
                        fig_vor = plot_pitch_control(
                            sel_ff, event_description=ev_desc,
                            match_label=sel["label"],
                            team_a_name=sel_team, team_b_name=opp_team,
                        )
                        st.plotly_chart(fig_vor, use_container_width=True, config=_PLOT_CFG)

                        if sel_ff:
                            with st.expander("📋 Freeze frame data table"):
                                ff_table = pd.DataFrame(sel_ff)
                                ff_table["team"] = ff_table["teammate"].map(
                                    {True: sel_team, False: opp_team})
                                st.dataframe(ff_table, use_container_width=True)
                    except Exception as e:
                        st.error(f"Pitch control error: {e}")


    #EXPECTED THREAT 
    with tab_xt:
        st.subheader(f"Expected Threat (xT) — {sel_team}  {min_label}")
        st.markdown(
            "*12×8 grid: brighter zones = higher goal probability. "
            "Arrows = most progressive pass/carry actions by the selected team.*"
        )

        with st.spinner("Building Markov Chain xT model…"):
            try:
                T, p_shoot, p_goal = build_transition_matrix(fdf)
                xT_grid = solve_xt(T, p_shoot, p_goal)

                col_xt1, col_xt2 = st.columns(2)

                with col_xt1:
                    st.markdown("#### xT Grid Heatmap")
                    fig_xt = plot_xt_grid(xT_grid, title=f"xT Grid  {min_label}")
                    st.pyplot(fig_xt, use_container_width=True)
                    st.metric("Peak xT (near goal)", f"{float(xT_grid.max()):.4f}")
                    st.metric("Mean xT (all zones)",  f"{float(xT_grid.mean()):.4f}")

                with col_xt2:
                    st.markdown("#### Top Threat-Added Actions")
                    top_n = st.slider("Show top N actions", 5, 50, 20, key="xt_n")
                    threat_df = compute_threat_added(fdf, xT_grid, sel_team)

                    fig_th = plot_threat_added(threat_df, sel_team, top_n=top_n)
                    st.pyplot(fig_th, use_container_width=True)

                    if not threat_df.empty and threat_df["xT_added"].gt(0).any():
                        st.markdown("#### Player xT Leaderboard")
                        player_xt = (
                            threat_df.groupby("player_name")["xT_added"]
                            .sum().reset_index()
                            .sort_values("xT_added", ascending=False)
                            .head(10)
                        )
                        player_xt.columns = ["Player", "Total xT Added"]
                        player_xt["Total xT Added"] = player_xt["Total xT Added"].round(4)
                        st.dataframe(player_xt, use_container_width=True, hide_index=True)
                    else:
                        st.info(
                            "No positive threat-added actions found for "
                            f"**{sel_team}** in this time window. "
                            "Try widening the minute range or selecting a different team."
                        )

            except Exception as e:
                st.error(f"xT model error: {e}")
                st.caption("Occurs if the selected window has too few events.")


    # TAB 5: SHOT MAP
    with tab_shots:
        st.subheader(f"Shot Map  {min_label}")
        st.markdown(
            "*Bubble size = xG · Color = outcome · Hover for full shot detail*"
        )

        shots = fdf[
            (fdf["event_type"] == "Shot") &
            fdf["x"].notna() & fdf["y"].notna()
        ].copy()

        if shots.empty:
            st.info("No shots recorded in this time window.")
        else:
            fig_shots = _base_pitch_figure(
                title="", height=580,
            )

            outcome_colors = {
                "Goal"   : "#ffdd00",
                "Saved"  : "#66fcf1",
                "Off T"  : "#ff4444",
                "Blocked": "#ff8c00",
                "Post"   : "#cc00cc",
                "Wayward": "#445566",
            }

            # Team color for team label (not outcome)
            team_color_map = {}
            all_shot_teams = shots["team_name"].unique().tolist()
            for i, tn in enumerate(all_shot_teams):
                team_color_map[tn] = CYAN if tn == sel_team else PINK

            for team in all_shot_teams:
                ts = shots[shots["team_name"] == team]

                # xG bubble sizes — CAPPED at 55px to avoid giant circles
                # Previously: xg * 400 + 80 → 393px for xG=0.78 → the yellow circle
                xg_vals = ts["statsbomb_xg"].fillna(0.05).clip(lower=0.02, upper=1.0)
                sizes   = (xg_vals * 45 + 10).clip(upper=55).tolist()   # max 55px

                colors_list = [outcome_colors.get(str(r.result), "#888")
                               for _, r in ts.iterrows()]

                hover = [
                    f"<b>{r.player_name}</b> ({r.team_name})<br>"
                    f"Minute: {r.minute}'{int(r.second or 0):02d}\"<br>"
                    f"Outcome: <b>{r.result}</b><br>"
                    f"xG: {r.statsbomb_xg:.3f}"
                    for _, r in ts.iterrows()
                ]

                # Subtle glow ring
                fig_shots.add_trace(go.Scatter(
                    x=ts["x"].tolist(), y=ts["y"].tolist(),
                    mode="markers",
                    marker=dict(size=[s + 10 for s in sizes],
                                color=colors_list, opacity=0.18, line=dict(width=0)),
                    showlegend=False, hoverinfo="none",
                ))

                # Main bubbles
                fig_shots.add_trace(go.Scatter(
                    x=ts["x"].tolist(), y=ts["y"].tolist(),
                    mode="markers",
                    marker=dict(size=sizes, color=colors_list, opacity=0.92,
                                line=dict(color=PITCH_BG, width=1.5)),
                    name=team,
                    hovertext=hover, hoverinfo="text",
                ))

            # Team xG annotations (top of pitch, inside figure)
            for i, team in enumerate(all_shot_teams):
                ts     = shots[shots["team_name"] == team]
                t_xg   = ts["statsbomb_xg"].sum()
                t_goal = int((ts["result"] == "Goal").sum())
                col    = CYAN if team == sel_team else PINK
                fig_shots.add_annotation(
                    x=5 + i * 58, y=76,
                    text=f"<b>{team}</b>: {t_goal}G  xG {t_xg:.2f}",
                    font=dict(color=col, size=11, family="Courier New"),
                    showarrow=False, xanchor="left",
                )

            # Outcome legend
            for outcome, color in outcome_colors.items():
                fig_shots.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=color),
                    name=outcome, showlegend=True,
                ))

            st.plotly_chart(fig_shots, use_container_width=True, config=_PLOT_CFG)

            # Summary table
            summary = (
                shots.groupby(["team_name","result"])
                .agg(count=("event_id","count"), total_xg=("statsbomb_xg","sum"))
                .round(3).reset_index()
            )
            summary.columns = ["Team","Outcome","Count","Total xG"]
            st.dataframe(summary, use_container_width=True, hide_index=True)


    # EVENTS
    with tab_raw:
        st.subheader("Events Explorer")

        fa, fb, fc = st.columns(3)
        with fa:
            all_types   = ["All"] + sorted(events_df["event_type"].unique().tolist())
            filt_type   = st.selectbox("Event type", all_types, key="ex_type")
        with fb:
            all_teams_  = ["All"] + teams
            filt_team   = st.selectbox("Team", all_teams_, key="ex_team")
        with fc:
            all_players = ["All"] + sorted(events_df["player_name"].dropna().unique().tolist())
            filt_player = st.selectbox("Player", all_players[:200], key="ex_player")

        vis = fdf.copy()
        if filt_type   != "All": vis = vis[vis["event_type"]   == filt_type]
        if filt_team   != "All": vis = vis[vis["team_name"]    == filt_team]
        if filt_player != "All": vis = vis[vis["player_name"]  == filt_player]

        st.markdown(f"**{len(vis):,} events** matching filters")

        # Event location scatter
        has_loc = vis.dropna(subset=["x","y"])
        if not has_loc.empty:
            fig_ev = _base_pitch_figure(title="", height=420)
            for team in has_loc["team_name"].unique():
                sub = has_loc[has_loc["team_name"] == team]
                color = CYAN if team == sel_team else PINK
                hover = [
                    f"<b>{r.player_name}</b> ({r.team_name})<br>"
                    f"Min {r.minute}:{int(r.second or 0):02d}  |  {r.event_type}<br>"
                    f"Result: {r.result if pd.notna(r.result) else 'N/A'}"
                    for _, r in sub.iterrows()
                ]
                fig_ev.add_trace(go.Scatter(
                    x=sub["x"].tolist(), y=sub["y"].tolist(),
                    mode="markers",
                    marker=dict(size=7, color=color, opacity=0.80,
                                line=dict(width=0.5, color=PITCH_BG)),
                    name=team, hovertext=hover, hoverinfo="text",
                ))
            st.plotly_chart(fig_ev, use_container_width=True, config=_PLOT_CFG)

        # Raw table — replace None with N/A for clarity
        disp_cols = [c for c in ["minute","second","period","event_type",
                                  "team_name","player_name","x","y",
                                  "end_x","end_y","result"]
                     if c in vis.columns]
        display   = vis[disp_cols].head(500).reset_index(drop=True)
        display   = display.fillna("N/A")   # explicit N/A instead of blank/None

        st.dataframe(display, use_container_width=True, height=380)
        if len(vis) > 500:
            st.caption(f"Showing first 500 of {len(vis):,} events.")

else:
    # PLAYER STATS PAGE
    render_player_stats_page()
