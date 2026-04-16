import os
from dotenv import load_dotenv

load_dotenv()

GOLF_API_KEY = os.getenv("GOLF_API_KEY", "")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "")

DEFAULT_RANDOM_STATE = 42
DEFAULT_TARGET_COLUMN = "round_score"

COURSE_COORDS = {
    "Augusta National": (33.5031, -82.0197),
    "Pebble Beach": (36.5684, -121.9474),
    "TPC Sawgrass": (30.1975, -81.3967),
}

TOURNAMENT_TO_COURSE = {
    "Masters Tournament": "Augusta National",
    "AT&T Pebble Beach Pro-Am": "Pebble Beach",
    "THE PLAYERS Championship": "TPC Sawgrass",
}

CUT_RULES = {
    "masters": {
        "top_n": 50,
        "ties": True,
        "within_leader_strokes": 10,
    },
    "us_open": {
        "top_n": 60,
        "ties": True,
        "within_leader_strokes": None,
    },
    "the_open": {
        "top_n": 70,
        "ties": True,
        "within_leader_strokes": None,
    },
    "open_championship": {
        "top_n": 70,
        "ties": True,
        "within_leader_strokes": None,
    },
    "pga_championship": {
        "top_n": 70,
        "ties": True,
        "within_leader_strokes": None,
    },
    "default": {
        "top_n": 65,
        "ties": True,
        "within_leader_strokes": None,
    },
}


def normalize_tournament_key(tournament_name: str) -> str:
    return tournament_name.lower().replace(" ", "_").replace("-", "_")


def get_cut_rule(tournament_name: str) -> dict:
    key = normalize_tournament_key(tournament_name)

    for rule_key, rule in CUT_RULES.items():
        if rule_key != "default" and rule_key in key:
            return rule

    return CUT_RULES["default"]