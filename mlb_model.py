"""
MLB Monte Carlo Simulation Engine
==================================
Run 100,000 simulations per game for all today's confirmed-lineup MLB games.

Outputs per game:
  - Win %  (Home / Away)
  - Run Line projections
  - Total runs projection  (Total / Home / Away)
  - First-5-Innings (F5) projections  (Total / Home / Away)
  - Comparison vs live market lines (ML, Spread, O/U)

Data sources:
  - MLB Stats API  (statsapi Python wrapper)
  - The Odds API   (live ML / Runline / Totals)
  - Baseball Reference park factors  (scraped)
  - Rotowire weather page            (Selenium + BeautifulSoup)

Usage:
  1. Fill in your API keys in CONFIG below.
  2. Run:  python mlb_montecarlo.py
  3. Results are printed to stdout AND saved to mlb_projections_YYYYMMDD.csv

Requirements (pip install):
  statsapi requests numpy pandas selenium beautifulsoup4 lxml webdriver-manager
"""

CONFIG = {
    "ODDS_API_KEY":        "YOUR-API-KEY",        # the-odds-api.com key
    "ODDS_REGIONS":        "us",                        # us | uk | eu | au
    "ODDS_BOOKMAKER_PREF": "draftkings",                # preferred bookmaker for reference line
    # Park factors are scraped from https://parristaylor.xyz/mlb/park-run-factor (no key needed)
    "WEATHER_URL":         "https://www.rotowire.com/baseball/weather.php",
    "SIMULATIONS":         100_000,                     # Monte Carlo iterations
    "F5_INNINGS":          5,                           # First-N innings window
    "CURRENT_SEASON":      2026,
    "CHROME_DRIVER_PATH":  None,                        # None → auto-detect via webdriver-manager
    "OUTPUT_CSV":          True,
    "HEADLESS_BROWSER":    True,
}

import re
import sys
import json
import math
import time
import logging
import datetime
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import requests
import statsapi
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("mlb_mc")

# 1.  MLB STATS API  ─  schedule / lineup / pitcher / team stats

class MLBDataFetcher:
    """Fetch all needed MLB Stats API data."""

    SEASON = CONFIG["CURRENT_SEASON"]

    # ── schedule ──────────────────────────────────────────────────────────────
    def get_today_games(self) -> list[dict]:
        today = datetime.date.today().strftime("%m/%d/%Y")
        log.info(f"Fetching MLB schedule for {today}")
        schedule = statsapi.schedule(date=today, sportId=1)
        # Keep only Final, In Progress, Pre-Game, Scheduled, Warmup
        keep = {"Preview", "Pre-Game", "Warmup", "Scheduled", "In Progress", "Final"}
        return [g for g in schedule if g.get("status") in keep]

    # ── confirmed lineup  ─────────────────────────────────────────────────────
    def get_lineups(self, game_pk: int) -> dict:
        """
        Returns {
          'home': { 'batting_order': [...playerId], 'pitcher': playerId },
          'away': { 'batting_order': [...playerId], 'pitcher': playerId }
        }

        Pitcher detection uses three tiers in order of reliability:
          1. gameData.probablePitchers  – populated as soon as SP is announced
          2. boxscore starting pitcher  – available once game is live/final
          3. statsapi.schedule()        – home/away_probable_pitcher_id keys
        """
        try:
            data     = statsapi.get("game", {"gamePk": game_pk})
            game_data = data.get("gameData", {})
            live_data = data.get("liveData", {})
            boxscore  = live_data.get("boxscore", {})
            teams_box = boxscore.get("teams", {})

            # ── Tier 1: probablePitchers (most reliable pre-game source) ──────
            probable = game_data.get("probablePitchers", {})
            probable_ids = {
                "home": probable.get("home", {}).get("id"),
                "away": probable.get("away", {}).get("id"),
            }
            log.debug(f"probablePitchers → home={probable_ids['home']} away={probable_ids['away']}")

            # ── Tier 3: schedule fallback (fetch once, reuse for both sides) ──
            # Real statsapi schedule keys: 'home_probable_pitcher_id' etc.
            sched_ids = {"home": None, "away": None}
            if probable_ids["home"] is None or probable_ids["away"] is None:
                try:
                    sched = statsapi.schedule(game_id=game_pk)
                    if sched:
                        g = sched[0]
                        sched_ids["home"] = (g.get("home_probable_pitcher_id") or
                                             g.get("home_pitcher_id"))
                        sched_ids["away"] = (g.get("away_probable_pitcher_id") or
                                             g.get("away_pitcher_id"))
                        log.debug(f"Schedule fallback → home={sched_ids['home']} "
                                  f"away={sched_ids['away']}")
                except Exception as e:
                    log.debug(f"Schedule pitcher fallback failed: {e}")

            result = {}
            for side in ("home", "away"):
                team_box      = teams_box.get(side, {})
                players       = team_box.get("players", {})
                batting_order = team_box.get("battingOrder", [])
                lineup_ids    = [int(pid) for pid in batting_order]

                # ── Tier 1 ────────────────────────────────────────────────────
                pitcher_id = probable_ids.get(side)

                # ── Tier 2: boxscore (game live/final) ────────────────────────
                if pitcher_id is None:
                    for pdata in players.values():
                        all_pos   = pdata.get("allPositions", [])
                        pos_abbrs = [p.get("abbreviation", "") for p in all_pos]
                        is_pitcher = ("P" in pos_abbrs or
                                      pdata.get("position", {}).get("abbreviation") == "P")
                        status     = pdata.get("gameStatus", {})
                        is_starter = (status.get("isCurrentBatter") is False and
                                      status.get("isOnBench") is False)
                        ip_str     = (pdata.get("stats", {})
                                          .get("pitching", {})
                                          .get("inningsPitched", ""))
                        if is_pitcher and (is_starter or ip_str not in ("", None)):
                            pitcher_id = pdata["person"]["id"]
                            break

                # ── Tier 3 ────────────────────────────────────────────────────
                if pitcher_id is None:
                    pitcher_id = sched_ids.get(side)

                # log which tier resolved the pitcher
                tier = ("T1-probable" if probable_ids.get(side) is not None else
                        "T2-boxscore" if (pitcher_id is not None and
                                          sched_ids.get(side) != pitcher_id) else
                        "T3-schedule" if sched_ids.get(side) is not None else
                        "NONE")
                log.debug(f"  {side} SP resolved via {tier}: id={pitcher_id}")

                if pitcher_id is not None:
                    pitcher_id = int(pitcher_id)

                result[side] = {
                    "batting_order": lineup_ids,
                    "pitcher":   pitcher_id,
                    "team_id":   team_box.get("team", {}).get("id"),
                    "team_name": team_box.get("team", {}).get("name", ""),
                }
            return result
        except Exception as e:
            log.warning(f"Lineup fetch failed for gamePk {game_pk}: {e}")
            return {}

    # ── pitcher season stats ──────────────────────────────────────────────────
    def get_pitcher_stats(self, player_id: int) -> dict:
        """Returns ERA, WHIP, K/9, BB/9, HR/9, IP for current season."""
        defaults = {"name": f"id={player_id}", "era": 4.50, "whip": 1.30,
                    "k9": 8.5, "bb9": 3.2, "hr9": 1.2, "ip": 0, "fip": 4.50}
        if not player_id:
            return defaults
        try:
            # Get player name for logging
            person = statsapi.get("person", {"personId": player_id})
            name   = (person.get("people", [{}])[0]
                            .get("fullName", f"id={player_id}"))

            data  = statsapi.player_stat_data(player_id, group="pitching",
                                              type="season", sportId=1)
            stats = data.get("stats", [{}])[0].get("stats", {})
            era   = float(stats.get("era",  4.50) or 4.50)
            whip  = float(stats.get("whip", 1.30) or 1.30)
            ip    = float(stats.get("inningsPitched", 0) or 0)
            k9    = float(stats.get("strikeoutsPer9Inn", 8.5) or 8.5)
            bb9   = float(stats.get("walksPer9Inn", 3.2) or 3.2)
            hr9   = float(stats.get("homeRunsPer9", 1.2) or 1.2)

            # FIP = (13*HR + 3*BB - 2*K) / IP + FIP_constant (3.10 = league constant)
            ks  = float(stats.get("strikeOuts", 0) or 0)
            bbs = float(stats.get("baseOnBalls", 0) or 0)
            hrs = float(stats.get("homeRuns", 0) or 0)

            if ip < 5:
                # Too few innings to trust current-season ERA/FIP (opener, injury return,
                # or first start of year). Blend 80% career / 20% current to avoid
                # extreme values skewing projections.
                career = statsapi.player_stat_data(player_id, group="pitching",
                                                   type="career", sportId=1)
                cs = career.get("stats", [{}])[0].get("stats", {})
                c_era = float(cs.get("era",  4.50) or 4.50)
                c_ip  = float(cs.get("inningsPitched", 1) or 1)
                c_ks  = float(cs.get("strikeOuts", 0) or 0)
                c_bbs = float(cs.get("baseOnBalls", 0) or 0)
                c_hrs = float(cs.get("homeRuns", 0) or 0)
                c_fip = ((13 * c_hrs + 3 * c_bbs - 2 * c_ks) / max(c_ip, 1)) + 3.10
                era  = 0.2 * era + 0.8 * c_era  if ip > 0 else c_era
                fip_val = 0.2 * ((13*hrs + 3*bbs - 2*ks) / max(ip,1) + 3.10)                           + 0.8 * c_fip          if ip > 0 else c_fip
                log.debug(f"  {name}: low IP={ip:.1f}, blending career ERA={c_era:.2f} → {era:.2f}")
            else:
                fip_val = ((13 * hrs + 3 * bbs - 2 * ks) / max(ip, 1)) + 3.10

            return {"name": name, "era": round(era, 2), "whip": whip,
                    "k9": k9, "bb9": bb9, "hr9": hr9,
                    "ip": ip, "fip": round(fip_val, 2)}
        except Exception as e:
            log.debug(f"Pitcher stats error (id={player_id}): {e}")
            return defaults

    # ── batter season stats ───────────────────────────────────────────────────
    def get_batter_stats(self, player_id: int) -> dict:
        """Returns wOBA proxy (OBP + SLG), BB%, K%, ISO for current season."""
        defaults = {"obp": 0.320, "slg": 0.420, "ops": 0.740,
                    "bb_pct": 0.085, "k_pct": 0.220, "iso": 0.155}
        if not player_id:
            return defaults
        try:
            data = statsapi.player_stat_data(player_id, group="hitting",
                                              type="season", sportId=1)
            stats = data.get("stats", [{}])[0].get("stats", {})
            obp  = float(stats.get("obp", 0.320) or 0.320)
            slg  = float(stats.get("slg", 0.420) or 0.420)
            ops  = float(stats.get("ops", 0.740) or 0.740)
            pa   = float(stats.get("plateAppearances", 1) or 1)
            bbs  = float(stats.get("baseOnBalls", 0) or 0)
            sos  = float(stats.get("strikeOuts", 0) or 0)
            ab   = float(stats.get("atBats", 1) or 1)
            hits = float(stats.get("hits", 0) or 0)
            hr   = float(stats.get("homeRuns", 0) or 0)
            singles = hits - float(stats.get("doubles", 0) or 0) \
                           - float(stats.get("triples", 0) or 0) - hr
            iso  = (singles * 1 + float(stats.get("doubles", 0) or 0) * 2
                    + float(stats.get("triples", 0) or 0) * 3
                    + hr * 4 - hits) / max(ab, 1)
            return {"obp": obp, "slg": slg, "ops": ops,
                    "bb_pct": bbs / max(pa, 1),
                    "k_pct":  sos / max(pa, 1),
                    "iso":    round(iso, 3)}
        except Exception as e:
            log.debug(f"Batter stats error (id={player_id}): {e}")
            return defaults

    # ── team season stats (O/U record proxy) ─────────────────────────────────
    def get_team_run_stats(self, team_id: int) -> dict:
        """Season runs/game averages + over/under record proxy."""
        defaults = {"runs_per_game_offense": 4.50,
                    "runs_per_game_allowed":  4.50,
                    "over_pct": 0.50}
        if not team_id:
            return defaults
        try:
            data = statsapi.get("team_stats", {
                "teamId": team_id, "group": "hitting",
                "stats": "season", "season": self.SEASON
            })
            splits = (data.get("stats", [{}])[0]
                          .get("splits", [{}])[0]
                          .get("stat", {}))
            games = float(splits.get("gamesPlayed", 162) or 162)
            runs  = float(splits.get("runs", 720) or 720)
            rpg   = runs / max(games, 1)

            pit_data = statsapi.get("team_stats", {
                "teamId": team_id, "group": "pitching",
                "stats": "season", "season": self.SEASON
            })
            pit_splits = (pit_data.get("stats", [{}])[0]
                                   .get("splits", [{}])[0]
                                   .get("stat", {}))
            runs_allowed = float(pit_splits.get("runs", 720) or 720)
            ra_pg = runs_allowed / max(games, 1)

            # naive O/U record: if team avg total (rpg + ra_pg) vs league avg ~9.1
            league_total = 9.10
            team_total   = rpg + ra_pg
            over_pct     = 0.50 + (team_total - league_total) * 0.03
            over_pct     = max(0.35, min(0.65, over_pct))

            return {"runs_per_game_offense": round(rpg, 2),
                    "runs_per_game_allowed":  round(ra_pg, 2),
                    "over_pct": round(over_pct, 3)}
        except Exception as e:
            log.debug(f"Team run stats error (id={team_id}): {e}")
            return defaults


# 2.  THE ODDS API  ─  ML / Runline / Total

class OddsAPIFetcher:
    BASE = "https://api.the-odds-api.com/v4"
    SPORT = "baseball_mlb"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache: Optional[list] = None

    def _fetch(self) -> list:
        if self._cache is not None:
            return self._cache
        url = f"{self.BASE}/sports/{self.SPORT}/odds"
        params = {
            "apiKey":     self.api_key,
            "regions":    CONFIG["ODDS_REGIONS"],
            "markets":    "h2h,spreads,totals",
            "oddsFormat": "american",
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            remaining = r.headers.get("x-requests-remaining", "?")
            log.info(f"Odds API quota remaining: {remaining}")
            self._cache = r.json()
            return self._cache
        except Exception as e:
            log.warning(f"Odds API fetch failed: {e}")
            return []

    def get_game_odds(self, home_team: str, away_team: str) -> dict:
        """
        Returns best-available market lines for ML, run line, and total.
        Fuzzy matches team names (last word of team name).
        """
        data = self._fetch()
        blank = {"ml_home": None, "ml_away": None,
                 "spread_home": None, "spread_away": None,
                 "spread_line": -1.5,
                 "total": None, "total_line": None,
                 "bookmaker": None}
        if not data:
            return blank

        def last_word(s): return s.split()[-1].lower()

        home_kw = last_word(home_team)
        away_kw = last_word(away_team)

        for event in data:
            if (last_word(event.get("home_team", "")) == home_kw and
                    last_word(event.get("away_team", "")) == away_kw):
                # prefer configured bookmaker, else take first
                books = event.get("bookmakers", [])
                bm = next((b for b in books
                           if b["key"] == CONFIG["ODDS_BOOKMAKER_PREF"]), None)
                if bm is None and books:
                    bm = books[0]
                if bm is None:
                    return blank

                result = dict(blank)
                result["bookmaker"] = bm["title"]
                for market in bm.get("markets", []):
                    key = market["key"]
                    outcomes = market.get("outcomes", [])
                    if key == "h2h":
                        for o in outcomes:
                            if last_word(o["name"]) == home_kw:
                                result["ml_home"] = o["price"]
                            else:
                                result["ml_away"] = o["price"]
                    elif key == "spreads":
                        for o in outcomes:
                            if last_word(o["name"]) == home_kw:
                                result["spread_home"] = o["price"]
                                result["spread_line"] = o.get("point", -1.5)
                            else:
                                result["spread_away"] = o["price"]
                    elif key == "totals":
                        for o in outcomes:
                            if o["name"].lower() == "over":
                                result["total"] = o["price"]
                                result["total_line"] = o.get("point", 9.0)
                return result

        log.debug(f"No odds match for {away_team} @ {home_team}")
        return blank

    @staticmethod
    def american_to_prob(american: Optional[int]) -> Optional[float]:
        """Convert American odds to implied probability (no vig removed)."""
        if american is None:
            return None
        if american > 0:
            return 100 / (american + 100)
        return abs(american) / (abs(american) + 100)


# 3.  PARK RUN FACTOR  ─  parristaylor.xyz/mlb/park-run-factor

class ParkFactorScraper:
    """
    Scrapes park run factors from https://parristaylor.xyz/mlb/park-run-factor.

    The page renders its data entirely via an inline <script> block containing:
        const parks = [
          { rk:1, team:'Rockies', abbr:'COL', venue:'Coors Field', prf:113 },
          ...
        ];
    No Selenium needed — requests + regex extracts all entries directly.

    Internal lookup table is keyed by three identifiers per park so that any
    MLB Stats API team name variant resolves correctly:
        • team nickname  e.g. "rockies"
        • abbreviation   e.g. "col"
        • venue keywords e.g. "coors", "field"
    """

    URL = "https://parristaylor.xyz/mlb/park-run-factor"

    # Regex: captures one JS object  { ..., team:'...', abbr:'...', venue:'...', prf:NNN }
    _ENTRY_RE = re.compile(
        r"\{\s*rk\s*:\s*\d+\s*,"      # rk:N
        r"\s*team\s*:\s*'([^']+)'"     # team:'...'   → group 1
        r".*?abbr\s*:\s*'([^']+)'"     # abbr:'...'   → group 2
        r".*?venue\s*:\s*'([^']+)'"    # venue:'...'  → group 3
        r".*?prf\s*:\s*(\d+)"          # prf:NNN      → group 4
        r"\s*\}",
        re.DOTALL,
    )

    def __init__(self):
        self._lookup: dict[str, float] = {}
        # Full records for logging
        self._records: list[dict] = []

    # Hardcoded fallback PRF table — used if the live page is unreachable.
    # Update these values whenever parristaylor.xyz/mlb/park-run-factor changes.
    _FALLBACK_PRF: dict[str, float] = {
        # abbr → PRF  (also indexed by nickname words in _build_fallback_lookup)
        "COL": 113, "BOS": 104, "ARI": 103, "CIN": 103, "MIN": 102,
        "LAD": 101, "ATL": 101, "LAA": 101, "PHI": 101, "KC":  101,
        "WSH": 101, "MIA": 101, "NYY": 100, "HOU": 100, "BAL": 100,
        "DET": 100, "STL": 100, "TOR": 100, "PIT":  99, "CWS":  99,
        "NYM":  98, "SD":   97, "MIL":  97, "CHC":  97, "SF":   97,
        "TEX":  97, "CLE":  97, "SEA":  91, "OAK": 100, "TB":  100,
    }
    _FULL_NAME_TO_ABBR: dict[str, str] = {
        "rockies": "COL", "sox": "BOS", "red": "BOS",
        "diamondbacks": "ARI", "dbacks": "ARI", "backs": "ARI",
        "reds": "CIN", "twins": "MIN", "dodgers": "LAD",
        "braves": "ATL", "angels": "LAA", "phillies": "PHI",
        "royals": "KC",  "nationals": "WSH", "marlins": "MIA",
        "yankees": "NYY", "astros": "HOU", "orioles": "BAL",
        "tigers": "DET", "cardinals": "STL", "jays": "TOR",
        "pirates": "PIT", "white": "CWS", "mets": "NYM",
        "padres": "SD",  "brewers": "MIL", "cubs": "CHC",
        "giants": "SF",  "rangers": "TEX", "guardians": "CLE",
        "mariners": "SEA", "athletics": "OAK", "rays": "TB",
    }

    def fetch(self):
        """Fetch the live page; fall back to hardcoded table if unreachable."""
        try:
            r = requests.get(
                self.URL,
                timeout=15,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
            )
            r.raise_for_status()
            self._parse(r.text)
            if self._records:
                log.info(f"Park factors loaded: {len(self._records)} parks (live scrape)")
                log.debug("Sample PRFs: " + ", ".join(
                    f"{p['team']} {p['prf']}" for p in self._records[:5]
                ))
            else:
                log.warning("Park factor scrape: page returned but no parks parsed "
                            "→ using hardcoded fallback table")
                self._load_fallback()
        except Exception as exc:
            log.warning(f"Park factor fetch failed: {exc}  →  using hardcoded fallback table")
            self._load_fallback()

    def _load_fallback(self):
        """Populate lookup from the hardcoded _FALLBACK_PRF table."""
        for abbr, prf in self._FALLBACK_PRF.items():
            self._lookup[abbr.lower()] = prf
        for keyword, abbr in self._FULL_NAME_TO_ABBR.items():
            self._lookup[keyword] = self._FALLBACK_PRF.get(abbr, 100.0)
        log.info(f"Park factors loaded: {len(self._lookup)} tokens (fallback table)")

    def get(self, team_name: str, venue_name: str = "") -> float:
        """
        Return PRF for the home team/venue.  Falls back to 100.0 (neutral).

        Matching priority:
          1. Any word in team_name against lookup tokens
          2. Any word in venue_name against lookup tokens
          3. 100.0 neutral default
        """
        for source in (team_name, venue_name):
            for word in source.lower().split():
                if len(word) >= 3 and word in self._lookup:
                    return self._lookup[word]
        return 100.0

    # ── internal parser ───────────────────────────────────────────────────────
    def _parse(self, html: str):
        """
        Extract every { rk, team, abbr, venue, prf } object from the inline
        <script> block and index by all useful tokens.
        """
        soup = BeautifulSoup(html, "lxml")
        script_text = ""
        for tag in soup.find_all("script"):
            if "const parks" in tag.get_text():
                script_text = tag.get_text()
                break

        if not script_text:
            log.warning("Park factors: 'const parks' block not found in page.")
            return

        for m in self._ENTRY_RE.finditer(script_text):
            team_raw  = m.group(1).strip()   # e.g. "Red Sox"
            abbr_raw  = m.group(2).strip()   # e.g. "BOS"
            venue_raw = m.group(3).strip()   # e.g. "Fenway Park"
            prf       = float(m.group(4))

            record = {"team": team_raw, "abbr": abbr_raw,
                      "venue": venue_raw, "prf": prf}
            self._records.append(record)

            # Index every useful token from team name, abbreviation, and venue
            tokens = set()
            tokens.add(abbr_raw.lower())                          # "bos"
            for word in team_raw.lower().split():                 # "red", "sox"
                if len(word) >= 3:
                    tokens.add(word)
            for word in venue_raw.lower().split():                # "fenway", "park"
                if len(word) >= 3 and word not in {"park", "field",
                                                    "stadium", "centre",
                                                    "center", "great"}:
                    tokens.add(word)

            for token in tokens:
                self._lookup[token] = prf


# 4.  WEATHER SCRAPER  ─  Rotowire (Selenium + BeautifulSoup)

class WeatherScraper:
    """
    Scrapes Rotowire weather page for wind speed/direction and temperature.
    Returns modifiers per game venue.

    Wind direction values stored:
      "in"   – blowing in (suppresses offense)
      "out"  – blowing out (boosts offense)
      "lr"   – L to R crosswind (minor boost)
      "rl"   – R to L crosswind (minor boost)
      "calm" – no meaningful wind
    """

    # Anchored to the MPH value so city/team names containing "in" don't fire.
    # Matches: "In from CF", "Out to CF", "L to R", "R to L", "Calm"
    _WIND_DIR_RE = re.compile(
        r'\b\d+\s*mph\s+(in\b|out\b|l\s+to\s+r|r\s+to\s+l|calm)',
        re.IGNORECASE,
    )

    def __init__(self):
        self._data: dict[str, dict] = {}   # keyed by team city/name fragment

    def fetch(self):
        url = CONFIG["WEATHER_URL"]
        html = self._get_html(url)
        if html:
            self._parse(html)
            log.info(f"Weather data loaded: {len(self._data)} venues")
        else:
            log.warning("Weather scrape failed. Wind/temp modifiers set to neutral.")

    # ── Selenium fetch ────────────────────────────────────────────────────────
    def _get_html(self, url: str) -> Optional[str]:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.common.by import By
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service

            opts = Options()
            if CONFIG["HEADLESS_BROWSER"]:
                opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument("--disable-gpu")
            opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 Chrome/120 Safari/537.36")

            driver_path = CONFIG["CHROME_DRIVER_PATH"]
            service = (Service(driver_path) if driver_path
                       else Service(ChromeDriverManager().install()))
            driver = webdriver.Chrome(service=service, options=opts)
            driver.get(url)
            # wait for weather cards to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "weather-box"))
                )
            except Exception:
                time.sleep(4)  # fallback wait
            html = driver.page_source
            driver.quit()
            return html
        except ImportError:
            log.warning("Selenium / webdriver-manager not installed. "
                        "Install: pip install selenium webdriver-manager")
            return None
        except Exception as e:
            log.warning(f"Selenium weather scrape error: {e}")
            return None

    def _parse(self, html: str):
        """
        Rotowire weather page structure (2024-25): each game has a .weather-box
        containing team names, temperature, wind speed, wind direction.
        Adapt CSS selectors below if Rotowire redesigns the page.
        """
        soup = BeautifulSoup(html, "lxml")
        boxes = soup.find_all(class_="weather-box")
        if not boxes:
            # try alternate structure
            boxes = soup.find_all("div", {"class": lambda c: c and "weather" in c})

        for box in boxes:
            text = box.get_text(" ", strip=True).lower()

            # ── temperature ───
            temp = 72  # default
            t_match = re.search(r"(\d{2,3})\s*°?f", text)
            if t_match:
                temp = int(t_match.group(1))

            # ── wind ─────────────────────────────────────────────────────────
            # Speed: first number followed by mph in the box text
            wind_mph = 0
            wind_dir = "calm"
            spd_m = re.search(r"(\d+)\s*mph", text)
            if spd_m:
                wind_mph = int(spd_m.group(1))

            # Direction: anchored to the mph value to avoid false matches
            # on team/city names that contain "in" (e.g. "wind", "twins", "cincinnati")
            # Rotowire direction strings: "In from CF/LF/RF", "Out to CF/LF/RF",
            #                            "L to R", "R to L", "Calm"
            dir_m = self._WIND_DIR_RE.search(text)
            if dir_m:
                raw_dir = dir_m.group(1).lower().strip()
                if raw_dir.startswith("in"):
                    wind_dir = "in"
                elif raw_dir.startswith("out"):
                    wind_dir = "out"
                elif "l to r" in raw_dir:
                    wind_dir = "lr"        # left-to-right crosswind
                elif "r to l" in raw_dir:
                    wind_dir = "rl"        # right-to-left crosswind
                else:
                    wind_dir = "calm"
            elif wind_mph == 0:
                wind_dir = "calm"

            # ── team identifier ─
            team_tags = box.find_all(class_=re.compile(r"team|matchup", re.I))
            teams_text = " ".join(t.get_text(strip=True) for t in team_tags)
            if not teams_text:
                teams_text = text[:60]

            entry = {
                "temp_f":    temp,
                "wind_mph":  wind_mph,
                "wind_dir":  wind_dir,
                "raw":       teams_text,
            }
            # store under each word of the teams text (>= 3 chars, consistent with park lookup)
            for word in teams_text.lower().split():
                if len(word) >= 3:
                    self._data[word] = entry

    def get(self, team_name: str) -> dict:
        """Return weather dict for a team. Defaults to neutral if not found."""
        neutral = {"temp_f": 72, "wind_mph": 0, "wind_dir": "calm"}
        for word in team_name.lower().split():
            if len(word) >= 3 and word in self._data:
                return self._data[word]
        return neutral

    @staticmethod
    def run_modifier(weather: dict) -> float:
        """
        Returns a run-scoring multiplier based on temperature and wind.
        Calibrated from historical park data:
          - Warm air (>75°F) slightly boosts offense
          - Wind blowing out boosts; in suppresses
        """
        temp   = weather.get("temp_f", 72)
        wind   = weather.get("wind_mph", 0)
        w_dir  = weather.get("wind_dir", "calm")

        # temperature effect (~0.4% per degree above/below 72°F)
        temp_mod = 1.0 + (temp - 72) * 0.004

        # "out" = blowing toward OF, boosts fly balls / carries
        # "in"  = blowing toward home plate, suppresses offense
        # "lr"/"rl" = crosswind, mild effect on foul territory / carry
        # "calm" or unknown = no adjustment
        if w_dir == "out":
            wind_mod = 1.0 + wind * 0.012
        elif w_dir == "in":
            wind_mod = 1.0 - wind * 0.010
        elif w_dir in ("lr", "rl"):
            wind_mod = 1.0 + wind * 0.003
        else:
            wind_mod = 1.0

        return max(0.70, min(1.45, temp_mod * wind_mod))


# 5.  SIMULATION ENGINE

class GameSimulator:
    """
    Core Monte Carlo engine.

    Scoring model:
      - Per-inning expected runs for each team derived from:
          * Batting lineup OPS/wOBA proxy (9-batter rolling through order)
          * Opposing pitcher ERA / FIP
          * Park run factor
          * Weather run modifier
          * Team O/U tendency weight
      - Innings 1-5 tracked separately for F5 projection
      - 9-inning game for full projection (extras handled via win-probability)
    """

    RNG = np.random.default_rng()

    # League average runs/inning ≈ 0.50 (9 inn × ~4.5 runs/game / 9 ≈ 0.50)
    LEAGUE_RUNS_PER_INNING = 0.505

    def __init__(self, n_sim: int = 100_000):
        self.n = n_sim

    # ── expected runs per inning ──────────────────────────────────────────────
    def _lineup_ops_factor(self, batter_stats: list[dict]) -> float:
        """
        Relative offensive strength of a 9-man lineup vs league average OPS.
        League avg OPS ≈ 0.715
        """
        if not batter_stats:
            return 1.0
        ops_vals = [b.get("ops", 0.715) for b in batter_stats[:9]]
        mean_ops = np.mean(ops_vals)
        return mean_ops / 0.715

    def _pitcher_era_factor(self, pitcher: dict) -> float:
        """
        Scale ERA to a run-suppression factor.
        League avg ERA ≈ 4.20  → factor = 1.0
        ERA 2.50 → factor ≈ 0.60  (suppressive)
        ERA 6.00 → factor ≈ 1.43  (permissive)
        """
        era = pitcher.get("era", 4.20)
        # blend ERA and FIP 50/50
        fip = pitcher.get("fip", era)
        blended = 0.5 * era + 0.5 * fip
        return blended / 4.20

    def expected_runs_per_inning(
        self,
        lineup_ops_factor:   float,
        pitcher_era_factor:  float,
        park_factor:         float,   # 100 = neutral
        weather_mod:         float,   # 1.0 = neutral
        team_over_pct:       float,   # 0.50 = neutral
    ) -> float:
        """
        Combine all factors into a single expected-runs-per-inning value.
        """
        # over_pct weight: teams that go over a lot → slight positive
        over_mod = 0.90 + team_over_pct * 0.20   # range 0.90–1.10

        xr = (self.LEAGUE_RUNS_PER_INNING
              * lineup_ops_factor
              * pitcher_era_factor
              * (park_factor / 100.0)
              * weather_mod
              * over_mod)
        return max(0.10, xr)

    # ── simulate one game ─────────────────────────────────────────────────────
    def simulate(
        self,
        home_xrpi: float,   # expected runs/inning for home offense
        away_xrpi: float,   # expected runs/inning for away offense
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorised simulation of n games using Poisson scoring.

        Returns (home_full, away_full, home_f5, away_f5)
        Each is shape (n,) int32 array of run totals.
        """
        n = self.n
        f5 = CONFIG["F5_INNINGS"]

        # F5 runs: sum of Poisson draws for 5 innings
        home_f5_runs = self.RNG.poisson(home_xrpi, size=(n, f5)).sum(axis=1)
        away_f5_runs = self.RNG.poisson(away_xrpi, size=(n, f5)).sum(axis=1)

        # Innings 6-9 for remaining full game
        home_late = self.RNG.poisson(home_xrpi, size=(n, 9 - f5)).sum(axis=1)
        away_late = self.RNG.poisson(away_xrpi, size=(n, 9 - f5)).sum(axis=1)

        home_full = home_f5_runs + home_late
        away_full = away_f5_runs + away_late

        # Tie-breaker: play extra innings until no ties remain (max 9 extras).
        # Each extra inning: Poisson draw with ghost-runner bump (~5% more offense).
        # We iterate inning-by-inning so ties can resolve at different innings.
        tie_mask = home_full == away_full
        max_extras = 9
        for _xi in range(max_extras):
            n_ties = int(tie_mask.sum())
            if n_ties == 0:
                break
            xi_h = self.RNG.poisson(home_xrpi * 1.05, size=n_ties)
            xi_a = self.RNG.poisson(away_xrpi * 1.05, size=n_ties)
            home_full[tie_mask] += xi_h
            away_full[tie_mask] += xi_a
            # Re-evaluate which games are still tied after this extra inning
            tie_mask = home_full == away_full

        return (home_full.astype(np.int32), away_full.astype(np.int32),
                home_f5_runs.astype(np.int32), away_f5_runs.astype(np.int32))

    # ── aggregate results ─────────────────────────────────────────────────────
    def aggregate(
        self,
        home_full: np.ndarray, away_full: np.ndarray,
        home_f5:   np.ndarray, away_f5:   np.ndarray,
    ) -> dict:
        n = self.n
        home_wins = (home_full > away_full).sum()
        away_wins = n - home_wins

        total_full = home_full + away_full
        total_f5   = home_f5   + away_f5

        return {
            # Win probabilities
            "Home_Win_Pct":  round(home_wins / n * 100, 2),
            "Away_Win_Pct":  round(away_wins / n * 100, 2),

            # Full game totals
            "Total_Projection":      round(total_full.mean(), 2),
            "Home_Total_Projection": round(home_full.mean(),  2),
            "Away_Total_Projection": round(away_full.mean(),  2),

            # F5 totals
            "F5_Total_Projection":      round(total_f5.mean(), 2),
            "F5_Home_Total_Projection": round(home_f5.mean(),  2),
            "F5_Away_Total_Projection": round(away_f5.mean(),  2),

            # Run line (1.5) cover %
            "Home_Covers_RL_Pct": round(((home_full - away_full) >= 1.5).sum() / n * 100, 2),
            "Away_Covers_RL_Pct": round(((away_full - home_full) >= 1.5).sum() / n * 100, 2),

            # Percentile ranges
            "Total_10th":  int(np.percentile(total_full, 10)),
            "Total_50th":  int(np.percentile(total_full, 50)),
            "Total_90th":  int(np.percentile(total_full, 90)),
            "F5_Total_10th": int(np.percentile(total_f5, 10)),
            "F5_Total_50th": int(np.percentile(total_f5, 50)),
            "F5_Total_90th": int(np.percentile(total_f5, 90)),
        }


# 6.  EDGE CALCULATOR  (model vs. market)

def calc_edge(sim: dict, odds: dict) -> dict:
    """
    Compare model win probability and total to market lines.
    Positive edge = model likes this side MORE than market does.
    """
    edges = {}

    # ML edge
    mkt_home = OddsAPIFetcher.american_to_prob(odds.get("ml_home"))
    mkt_away = OddsAPIFetcher.american_to_prob(odds.get("ml_away"))
    if mkt_home:
        edges["ML_Home_Edge_Pct"] = round((sim["Home_Win_Pct"] / 100 - mkt_home) * 100, 2)
    if mkt_away:
        edges["ML_Away_Edge_Pct"] = round((sim["Away_Win_Pct"] / 100 - mkt_away) * 100, 2)

    # Total edge
    mkt_total = odds.get("total_line")
    if mkt_total:
        proj_total = sim["Total_Projection"]
        edges["Total_Edge"]       = round(proj_total - mkt_total, 2)  # + = lean Over
        edges["Total_Line_Market"]= mkt_total
        edges["Total_Proj"]       = proj_total

    # Run line
    rl_line = odds.get("spread_line", -1.5)  # typical -1.5 for home fav
    if rl_line is not None:
        # model cover pct vs 50% break-even
        home_rl_cover = sim["Home_Covers_RL_Pct"]
        mkt_rl_prob   = OddsAPIFetcher.american_to_prob(odds.get("spread_home")) or 0.50
        edges["RL_Home_Edge_Pct"] = round((home_rl_cover / 100 - mkt_rl_prob) * 100, 2)

    return edges


# 7.  MAIN ORCHESTRATOR

def main():
    log.info("=" * 70)
    log.info("MLB Monte Carlo Simulation Engine  –  100,000 iterations/game")
    log.info(f"Date: {datetime.date.today()}")
    log.info("=" * 70)

    mlb      = MLBDataFetcher()
    odds_api = OddsAPIFetcher(CONFIG["ODDS_API_KEY"])
    parks    = ParkFactorScraper()
    weather  = WeatherScraper()
    sim_eng  = GameSimulator(n_sim=CONFIG["SIMULATIONS"])

    parks.fetch()
    weather.fetch()

    games = mlb.get_today_games()
    if not games:
        log.warning("No MLB games found for today. Exiting.")
        sys.exit(0)

    log.info(f"Found {len(games)} game(s) on today's schedule")

    all_results = []

    for game in games:
        game_pk   = game["game_id"]
        home_name = game.get("home_name", "Home")
        away_name = game.get("away_name", "Away")
        venue     = game.get("venue_name", "")
        game_time = game.get("game_datetime", "")

        log.info(f"\n{'─'*60}")
        log.info(f"  {away_name}  @  {home_name}  ({game_time})")
        log.info(f"  Venue: {venue}  |  gamePk: {game_pk}")

        # ── lineups ─────────────────────────────────────────────────────────
        lineups = mlb.get_lineups(game_pk)
        if not lineups:
            log.warning("  ⚠ Could not fetch lineups – skipping game.")
            continue

        home_lu = lineups.get("home", {})
        away_lu = lineups.get("away", {})

        home_pitcher_id = home_lu.get("pitcher")
        away_pitcher_id = away_lu.get("pitcher")
        home_batting    = home_lu.get("batting_order", [])
        away_batting    = away_lu.get("batting_order", [])
        home_team_id    = home_lu.get("team_id")
        away_team_id    = away_lu.get("team_id")

        # ── pitcher stats ────────────────────────────────────────────────────
        home_sp = mlb.get_pitcher_stats(home_pitcher_id)
        away_sp = mlb.get_pitcher_stats(away_pitcher_id)

        log.info(f"  Home SP: {home_sp.get('name', 'Unknown')} (id={home_pitcher_id})  "
                 f"ERA={home_sp['era']}  FIP={home_sp['fip']}")
        log.info(f"  Away SP: {away_sp.get('name', 'Unknown')} (id={away_pitcher_id})  "
                 f"ERA={away_sp['era']}  FIP={away_sp['fip']}")
        log.info(f"  Home lineup: {len(home_batting)} batters  |  "
                 f"Away lineup: {len(away_batting)} batters")

        # ── batter stats (lineup order) ──────────────────────────────────────
        home_batters = [mlb.get_batter_stats(pid) for pid in home_batting[:9]]
        away_batters = [mlb.get_batter_stats(pid) for pid in away_batting[:9]]
        if not home_batters:
            home_batters = [mlb.get_batter_stats(None)]
        if not away_batters:
            away_batters = [mlb.get_batter_stats(None)]

        # ── team run stats ───────────────────────────────────────────────────
        home_team_stats = mlb.get_team_run_stats(home_team_id)
        away_team_stats = mlb.get_team_run_stats(away_team_id)

        # ── park factor ──────────────────────────────────────────────────────
        park_factor = parks.get(home_name, venue)

        # ── weather ──────────────────────────────────────────────────────────
        wx       = weather.get(home_name)
        wx_mod   = WeatherScraper.run_modifier(wx)
        log.info(f"  Weather: {wx['temp_f']}°F  wind {wx['wind_mph']} mph "
                 f"{wx['wind_dir']}  →  run modifier {wx_mod:.3f}")

        # ── market odds ──────────────────────────────────────────────────────
        mkt = odds_api.get_game_odds(home_name, away_name)
        log.info(f"  Odds ({mkt.get('bookmaker','N/A')}): "
                 f"ML H={mkt.get('ml_home')} A={mkt.get('ml_away')}  "
                 f"Total={mkt.get('total_line')}  "
                 f"Spread={mkt.get('spread_line')}")

        # ── expected runs / inning ───────────────────────────────────────────
        # Away offense vs HOME pitcher; Home offense vs AWAY pitcher
        home_xrpi = sim_eng.expected_runs_per_inning(
            lineup_ops_factor  = sim_eng._lineup_ops_factor(home_batters),
            pitcher_era_factor = sim_eng._pitcher_era_factor(away_sp),   # away SP faces home batters
            park_factor        = park_factor,
            weather_mod        = wx_mod,
            team_over_pct      = home_team_stats["over_pct"],
        )
        away_xrpi = sim_eng.expected_runs_per_inning(
            lineup_ops_factor  = sim_eng._lineup_ops_factor(away_batters),
            pitcher_era_factor = sim_eng._pitcher_era_factor(home_sp),   # home SP faces away batters
            park_factor        = park_factor,
            weather_mod        = wx_mod,
            team_over_pct      = away_team_stats["over_pct"],
        )

        log.info(f"  xR/inn →  Home offense: {home_xrpi:.4f}  |  Away offense: {away_xrpi:.4f}")

        # ── Monte Carlo ──────────────────────────────────────────────────────
        log.info(f"  Running {CONFIG['SIMULATIONS']:,} simulations …")
        t0 = time.perf_counter()
        h_full, a_full, h_f5, a_f5 = sim_eng.simulate(home_xrpi, away_xrpi)
        elapsed = time.perf_counter() - t0
        log.info(f"  Simulation complete in {elapsed:.2f}s")

        results = sim_eng.aggregate(h_full, a_full, h_f5, a_f5)
        edges   = calc_edge(results, mkt)

        # ── build output row ─────────────────────────────────────────────────
        row = {
            "Date":       datetime.date.today().isoformat(),
            "Game":       f"{away_name} @ {home_name}",
            "GamePk":     game_pk,
            "Home":       home_name,
            "Away":       away_name,
            "Venue":      venue,
            "GameTime":   game_time,
            "Home_SP_ERA": home_sp["era"],
            "Home_SP_FIP": home_sp["fip"],
            "Away_SP_ERA": away_sp["era"],
            "Away_SP_FIP": away_sp["fip"],
            "ParkFactor":  park_factor,
            "WeatherMod":  round(wx_mod, 3),
            "WindMph":     wx["wind_mph"],
            "WindDir":     wx["wind_dir"],
            "TempF":       wx["temp_f"],
            **results,
            "Market_ML_Home":    mkt.get("ml_home"),
            "Market_ML_Away":    mkt.get("ml_away"),
            "Market_Spread":     mkt.get("spread_line"),
            "Market_Total":      mkt.get("total_line"),
            **edges,
        }
        all_results.append(row)

        # ── pretty print ─────────────────────────────────────────────────────
        _print_game_card(row)

    # ── save CSV ──────────────────────────────────────────────────────────────
    if all_results and CONFIG["OUTPUT_CSV"]:
        today_str = datetime.date.today().strftime("%Y%m%d")
        filename  = f"mlb_projections_{today_str}.csv"
        df = pd.DataFrame(all_results)
        df.to_csv(filename, index=False)
        log.info(f"\n✅  Results saved → {filename}")
    elif not all_results:
        log.warning("No games with confirmed lineups found today.")


# 8.  DISPLAY HELPER

def _print_game_card(row: dict):
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  {'AWAY':>25}  vs  {'HOME':<25}")
    print(f"  {row['Away']:>25}      {row['Home']:<25}")
    print(f"  {'ERA ':>25}  {row['Away_SP_ERA']:.2f}  /  {row['Home_SP_ERA']:.2f}")
    print(f"  {'FIP ':>25}  {row['Away_SP_FIP']:.2f}  /  {row['Home_SP_FIP']:.2f}")
    _dir_labels = {"in": "In", "out": "Out", "lr": "L→R", "rl": "R→L", "calm": "Calm"}
    wind_label = _dir_labels.get(row['WindDir'], row['WindDir'])
    print(f"  Park Factor: {row['ParkFactor']:.0f}  |  "
          f"Weather mod: {row['WeatherMod']:.3f}  "
          f"({row['TempF']}°F, {row['WindMph']} mph {wind_label})")
    print(f"{sep}")
    print(f"  {'WIN PROBABILITY':30}  Away {row['Away_Win_Pct']:5.1f}%  /  Home {row['Home_Win_Pct']:5.1f}%")
    print(f"  {'RUN LINE (-1.5) COVER %':30}  Away {row['Away_Covers_RL_Pct']:5.1f}%  /  Home {row['Home_Covers_RL_Pct']:5.1f}%")
    print(f"{sep}")
    print(f"  {'FULL GAME PROJECTIONS':30}")
    print(f"    Total Runs  : {row['Total_Projection']:.2f}  "
          f"(10th–90th: {row['Total_10th']}–{row['Total_90th']})")
    print(f"    Away Runs   : {row['Away_Total_Projection']:.2f}")
    print(f"    Home Runs   : {row['Home_Total_Projection']:.2f}")
    print(f"  {'FIRST 5 INNINGS':30}")
    print(f"    F5 Total    : {row['F5_Total_Projection']:.2f}  "
          f"(10th–90th: {row['F5_Total_10th']}–{row['F5_Total_90th']})")
    print(f"    F5 Away     : {row['F5_Away_Total_Projection']:.2f}")
    print(f"    F5 Home     : {row['F5_Home_Total_Projection']:.2f}")
    print(f"{sep}")
    print(f"  MARKET LINES ({row.get('Market_Total','N/A')} total  |  "
          f"ML H={row.get('Market_ML_Home','–')} A={row.get('Market_ML_Away','–')}  |  "
          f"Spread {row.get('Market_Spread','–')})")
    ml_he = row.get('ML_Home_Edge_Pct', 'N/A')
    ml_ae = row.get('ML_Away_Edge_Pct', 'N/A')
    tot_e = row.get('Total_Edge', 'N/A')
    rl_he = row.get('RL_Home_Edge_Pct', 'N/A')
    print(f"  EDGE vs MARKET:")
    print(f"    ML  Home {ml_he:+.2f}%  |  Away {ml_ae:+.2f}%" if isinstance(ml_he, float) else
          f"    ML  Home N/A  |  Away N/A")
    print(f"    Total Edge: {tot_e:+.2f} runs  (+ = lean OVER)" if isinstance(tot_e, float) else
          f"    Total Edge: N/A")
    print(f"    RL Home Edge: {rl_he:+.2f}%" if isinstance(rl_he, float) else
          f"    RL Home Edge: N/A")
    print(f"{sep}\n")

if __name__ == "__main__":
    main()
