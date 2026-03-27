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