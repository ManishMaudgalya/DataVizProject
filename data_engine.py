# data_engine.py  —  Pure StatsBomb JSON parser
# ROOT CAUSE FIX: kloppy's to_pandas() stores coordinates as Point objects
# in a single 'coordinates' column, NOT as 'x'/'y' float columns. This broke
# every downstream module. Solution: bypass kloppy entirely and parse the raw
# StatsBomb JSON ourselves. This gives us complete control over column names
# and guarantees:
#   x, y       → float, start location
#   end_x, end_y → float, end location (passes/carries/shots)
#   team_name  → str  (e.g. "Slovenia")
#   player_name → str (e.g. "Jan Oblak")
#   event_type → str  (e.g. "Pass", "Shot")
#   result     → str  (e.g. "Complete", "Goal")

import io
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

# CONSTANTS
BASE_URL       = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITION_ID = 55    # UEFA Euro 2024
SEASON_ID      = 282


# LOW-LEVEL HELPERS
def _fetch_json(url: str):
    """Stream a JSON file from GitHub. Raises HTTPError on 404/5xx."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _safe_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan



# STEP 1A  –  MATCH LIST

@st.cache_data(show_spinner="Loading Euro 2024 fixture list…")
def get_match_list() -> pd.DataFrame:
    """
    Return a DataFrame of all Euro 2024 matches.
    Columns: match_id, home_team, away_team, home_score, away_score,
             match_date, stage, label
    """
    raw = _fetch_json(f"{BASE_URL}/matches/{COMPETITION_ID}/{SEASON_ID}.json")

    rows = []
    for m in raw:
        ht = m["home_team"]["home_team_name"]
        at = m["away_team"]["away_team_name"]
        hs = m.get("home_score", 0)
        as_ = m.get("away_score", 0)
        rows.append({
            "match_id"  : m["match_id"],
            "home_team" : ht,
            "away_team" : at,
            "home_score": hs,
            "away_score": as_,
            "match_date": m.get("match_date", ""),
            "stage"     : m["competition_stage"]["name"],
            "label"     : f"{ht} {hs}–{as_} {at}  ({m['competition_stage']['name']})",
        })

    return pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)



# STEP 1B  –  PARSE RAW STATSBOMB EVENTS → CLEAN DATAFRAME

def _parse_statsbomb_events(raw_events: list) -> pd.DataFrame:
    """
    Convert a raw StatsBomb events list (from JSON) into a flat, typed DataFrame.

    StatsBomb JSON structure (simplified):
    {
      "id": "uuid",
      "type": {"name": "Pass"},
      "minute": 23, "second": 14, "period": 1,
      "team": {"id": 944, "name": "Slovenia"},
      "player": {"id": 31866, "name": "Jan Oblak"},
      "location": [x, y],            ← start of action
      "pass": {
          "end_location": [x2, y2],  ← end location
          "recipient": {"id": .., "name": ".."},
          "outcome": {"name": "Incomplete"}  ← null if complete
      },
      "carry": {"end_location": [x2, y2]},
      "shot": {"end_location": [x2, y2, z], "outcome": {"name": "Goal"},
               "statsbomb_xg": 0.12}
    }
    """
    rows = []
    for evt in raw_events:
        loc = evt.get("location") or []

        # --- Extract end_location and result based on event sub-type -------
        end_loc = []
        result  = None
        pass_recipient_id   = ""
        pass_recipient_name = ""
        statsbomb_xg = 0.0

        if "pass" in evt:
            p = evt["pass"]
            end_loc = p.get("end_location") or []
            outcome = p.get("outcome")
            # StatsBomb: no outcome dict → pass was complete
            result  = outcome["name"] if outcome else "Complete"
            rec = p.get("recipient") or {}
            pass_recipient_id   = str(rec.get("id", ""))
            pass_recipient_name = rec.get("name", "")

        elif "carry" in evt:
            end_loc = evt["carry"].get("end_location") or []
            result  = "Complete"

        elif "shot" in evt:
            s = evt["shot"]
            raw_end = s.get("end_location") or []
            end_loc = raw_end[:2]           # drop z-component if present
            result  = (s.get("outcome") or {}).get("name", "")
            statsbomb_xg = _safe_float(s.get("statsbomb_xg", 0))

        elif "dribble" in evt:
            result = (evt["dribble"].get("outcome") or {}).get("name", "")

        elif "interception" in evt:
            result = (evt["interception"].get("outcome") or {}).get("name", "")

        rows.append({
            "event_id"           : evt.get("id", ""),
            "index"              : int(evt.get("index", 0)),
            "period"             : int(evt.get("period", 1)),
            "timestamp"          : evt.get("timestamp", ""),
            "minute"             : int(evt.get("minute", 0)),
            "second"             : int(evt.get("second", 0)),
            # Clean string event type — no enum objects
            "event_type"         : evt.get("type", {}).get("name", "Unknown"),
            "team_id"            : str(evt.get("team", {}).get("id", "")),
            "team_name"          : evt.get("team", {}).get("name", ""),
            "player_id"          : str(evt.get("player", {}).get("id", "")),
            "player_name"        : evt.get("player", {}).get("name", ""),
            # Float coordinates — guaranteed, no Point objects
            "x"                  : _safe_float(loc[0]) if len(loc) >= 2 else np.nan,
            "y"                  : _safe_float(loc[1]) if len(loc) >= 2 else np.nan,
            "end_x"              : _safe_float(end_loc[0]) if len(end_loc) >= 2 else np.nan,
            "end_y"              : _safe_float(end_loc[1]) if len(end_loc) >= 2 else np.nan,
            "result"             : result,
            "pass_recipient_id"  : pass_recipient_id,
            "pass_recipient_name": pass_recipient_name,
            "statsbomb_xg"       : statsbomb_xg,
        })

    df = pd.DataFrame(rows)

    # Ensure numeric columns are correct dtype
    for col in ["x", "y", "end_x", "end_y", "statsbomb_xg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df



# STEP 1C  –  LOAD EVENTS (CACHED)

@st.cache_data(show_spinner="Streaming match events from StatsBomb…")
def get_match_events(match_id: int) -> pd.DataFrame:
    """
    Fetch and parse all events for a match.
    Returns a clean DataFrame — see _parse_statsbomb_events() for columns.
    """
    url = f"{BASE_URL}/events/{match_id}.json"
    raw = _fetch_json(url)
    return _parse_statsbomb_events(raw)



# STEP 1D  –  LOAD 360° FREEZE FRAMES (CACHED)

@st.cache_data(show_spinner="Loading 360° freeze frames…")
def get_freeze_frames(match_id: int) -> dict:
    """
    Parse three-sixty.json for a match.

    Returns dict: { event_id_str → list[dict] }
    Each player dict: {x, y, teammate, actor, keeper}

    StatsBomb 360 JSON structure:
    [{"id": "<event_uuid>",
      "freeze_frame": [
        {"location": [x, y], "teammate": bool, "actor": bool, "keeper": bool},
        ...
      ],
      "visible_area": [...]
    }, ...]
    """
    url = f"{BASE_URL}/three-sixty/{match_id}.json"
    try:
        raw = _fetch_json(url)
    except requests.HTTPError:
        return {}

    frames = {}
    for entry in raw:
        eid = entry.get("id", "")
        players = []
        for p in entry.get("freeze_frame", []):
            loc = p.get("location") or []
            if len(loc) < 2:
                continue
            players.append({
                "x"       : float(loc[0]),
                "y"       : float(loc[1]),
                "teammate": bool(p.get("teammate", False)),
                "actor"   : bool(p.get("actor", False)),
                "keeper"  : bool(p.get("keeper", False)),
            })
        if players:
            frames[eid] = players

    return frames



# STEP 1E  –  CONVENIENCE LOADER
def load_match_data(match_id: int) -> tuple:
    """Load and return (events_df, freeze_frames_dict) for a match."""
    return get_match_events(match_id), get_freeze_frames(match_id)



# UTILITY FUNCTIONS
def get_teams(df: pd.DataFrame) -> list:
    """Return list of unique team names present in the events DataFrame."""
    return sorted(df["team_name"].dropna().unique().tolist())


def get_players(df: pd.DataFrame, team_name: str) -> list:
    """Return sorted list of player names for a given team."""
    mask = df["team_name"] == team_name
    return sorted(df.loc[mask, "player_name"].dropna().unique().tolist())


def filter_by_time(df: pd.DataFrame, minute_min: int, minute_max: int) -> pd.DataFrame:
    """Slice events to a minute window (inclusive)."""
    return df[df["minute"].between(minute_min, minute_max)].copy()