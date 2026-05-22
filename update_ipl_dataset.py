"""
Build matches.csv / deliveries.csv for CricScope (IPL 2008-2025).

Sources:
  --source path/to/IPL.csv   Local Kaggle export (single ball-by-ball file)
  (default)                  Hugging Face mirror of chaitu20/ipl-dataset2008-2025
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ID = "prasad-gade05/ipl-enriched-2008-2025"

TEAM_ALIASES = {
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Pune Warriors": "Pune Warriors India",
}


def _normalize_teams(series: pd.Series) -> pd.Series:
    return series.replace(TEAM_ALIASES)


def _ipl_season_year(season: str | int | float) -> int:
    """Map Kaggle season labels (e.g. 2007/08, 2020/21) to IPL calendar year."""
    s = str(season).strip()
    if "/" in s:
        start, end = s.split("/", 1)
        start_y, end_y = int(start), int(end)
        if end_y < 100:
            end_y += (start_y // 100) * 100
        return end_y
    return int(float(s))


def _parse_win_margin(outcome) -> tuple[float, float]:
    if pd.isna(outcome):
        return 0.0, 0.0
    text = str(outcome).strip().lower()
    match = re.match(r"^(\d+(?:\.\d+)?)\s+(runs|wickets)$", text)
    if not match:
        return 0.0, 0.0
    value = float(match.group(1))
    if match.group(2) == "runs":
        return value, 0.0
    return 0.0, value


def _load_from_huggingface() -> tuple[pd.DataFrame, pd.DataFrame]:
    from huggingface_hub import hf_hub_download

    print("Downloading match_summary.parquet ...")
    match_path = hf_hub_download(repo_id=REPO_ID, filename="match_summary.parquet", repo_type="dataset")
    print("Downloading ball_by_ball.parquet ...")
    ball_path = hf_hub_download(repo_id=REPO_ID, filename="ball_by_ball.parquet", repo_type="dataset")
    return pd.read_parquet(match_path), pd.read_parquet(ball_path)


def _load_from_kaggle_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Reading {path} ...")
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[raw["innings"].isin([1, 2])].copy()

    inn1 = raw[raw["innings"] == 1].drop_duplicates(subset=["match_id"])
    meta = raw.groupby("match_id", as_index=False).first()

    win_margins = meta["win_outcome"].apply(_parse_win_margin)
    meta["win_margin_value"] = [m[0] or m[1] for m in win_margins]
    meta["win_margin_type"] = [
        "runs" if m[0] else ("wickets" if m[1] else np.nan) for m in win_margins
    ]
    meta["season"] = meta["season"].map(_ipl_season_year)
    meta["team1"] = inn1.set_index("match_id")["batting_team"].reindex(meta["match_id"]).values
    meta["team2"] = inn1.set_index("match_id")["bowling_team"].reindex(meta["match_id"]).values
    meta["result_type"] = meta["result_type"].fillna("normal")

    match_summary = meta[
        [
            "match_id",
            "date",
            "season",
            "venue",
            "city",
            "toss_winner",
            "toss_decision",
            "match_won_by",
            "win_margin_value",
            "win_margin_type",
            "player_of_match",
            "result_type",
            "method",
            "team1",
            "team2",
        ]
    ].copy()

    ball_by_ball = raw.rename(columns={"innings": "innings"})
    if "is_super_over" not in ball_by_ball.columns:
        ball_by_ball["is_super_over"] = 0

    return match_summary, ball_by_ball


def build_matches(match_summary: pd.DataFrame) -> pd.DataFrame:
    ms = match_summary.sort_values(["season", "date", "match_id"]).reset_index(drop=True)
    ms["id"] = np.arange(1, len(ms) + 1)

    win_runs = np.where(ms["win_margin_type"] == "runs", ms["win_margin_value"].fillna(0), 0)
    win_wkts = np.where(ms["win_margin_type"] == "wickets", ms["win_margin_value"].fillna(0), 0)

    dates = pd.to_datetime(ms["date"], errors="coerce")

    matches = pd.DataFrame(
        {
            "id": ms["id"].astype(int),
            "Season": "IPL-" + ms["season"].astype(str),
            "city": ms["city"],
            "date": dates.dt.strftime("%d-%m-%Y"),
            "team1": _normalize_teams(ms["team1"]),
            "team2": _normalize_teams(ms["team2"]),
            "toss_winner": _normalize_teams(ms["toss_winner"]),
            "toss_decision": ms["toss_decision"],
            "result": ms["result_type"],
            "dl_applied": (ms["method"] != "no_dls").astype(int),
            "winner": _normalize_teams(ms["match_won_by"]),
            "win_by_runs": win_runs,
            "win_by_wickets": win_wkts,
            "player_of_match": ms["player_of_match"],
            "venue": ms["venue"],
            "umpire1": np.nan,
            "umpire2": np.nan,
            "umpire3": np.nan,
        }
    )
    return matches


def build_deliveries(ball_by_ball: pd.DataFrame, id_map: dict) -> pd.DataFrame:
    balls = ball_by_ball[ball_by_ball["innings"].isin([1, 2])].copy()
    balls["match_id"] = balls["match_id"].map(id_map)
    balls = balls.dropna(subset=["match_id"])
    balls["match_id"] = balls["match_id"].astype(int)

    dismissed = balls["player_out"].where(
        balls["wicket_kind"].notna() & (balls["player_out"].astype(str) != "none"),
        np.nan,
    )

    deliveries = pd.DataFrame(
        {
            "match_id": balls["match_id"],
            "inning": balls["innings"].astype(int),
            "batting_team": _normalize_teams(balls["batting_team"]),
            "bowling_team": _normalize_teams(balls["bowling_team"]),
            "over": balls["over"].astype(int),
            "ball": balls["ball"].astype(int),
            "batsman": balls["batter"],
            "non_striker": balls["non_striker"],
            "bowler": balls["bowler"],
            "is_super_over": balls["is_super_over"].astype(int),
            "wide_runs": 0,
            "bye_runs": 0,
            "legbye_runs": 0,
            "noball_runs": 0,
            "penalty_runs": 0,
            "batsman_runs": balls["runs_batter"].fillna(0).astype(int),
            "extra_runs": balls["runs_extras"].fillna(0).astype(int),
            "total_runs": balls["runs_total"].fillna(0).astype(int),
            "player_dismissed": dismissed,
            "dismissal_kind": balls["wicket_kind"],
            "fielder": balls["fielders"],
        }
    )
    return deliveries


def main() -> None:
    parser = argparse.ArgumentParser(description="Update IPL CSV datasets (2008-2025)")
    parser.add_argument(
        "--source",
        type=Path,
        help="Local Kaggle IPL.csv (ball-by-ball export). Example: C:\\Users\\User\\Downloads\\IPL.csv",
    )
    parser.add_argument("--no-backup", action="store_true", help="Skip backing up existing CSVs")
    args = parser.parse_args()

    if args.source:
        match_summary, ball_by_ball = _load_from_kaggle_csv(args.source.resolve())
    else:
        match_summary, ball_by_ball = _load_from_huggingface()

    matches = build_matches(match_summary)
    ordered = match_summary.sort_values(["season", "date", "match_id"])
    id_map = dict(zip(ordered["match_id"], matches["id"]))
    deliveries = build_deliveries(ball_by_ball, id_map)

    if not args.no_backup:
        for name in ("matches.csv", "deliveries.csv"):
            src = ROOT / name
            if src.exists():
                dst = ROOT / f"{name}.bak"
                shutil.copy2(src, dst)
                print(f"Backed up {name} -> {name}.bak")

    matches.to_csv(ROOT / "matches.csv", index=False)
    deliveries.to_csv(ROOT / "deliveries.csv", index=False)

    seasons = sorted(matches["Season"].unique())
    print(f"Wrote matches.csv: {len(matches)} rows, seasons {seasons[0]} .. {seasons[-1]}")
    print(f"Wrote deliveries.csv: {len(deliveries)} rows, {deliveries['match_id'].nunique()} matches")
    teams = sorted(
        set(deliveries["batting_team"].unique()) | set(deliveries["bowling_team"].unique())
    )
    print(f"Teams in deliveries ({len(teams)}): {', '.join(teams)}")


if __name__ == "__main__":
    main()
