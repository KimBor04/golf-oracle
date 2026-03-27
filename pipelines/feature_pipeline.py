from src.paths import FEATURES_DIR, LIVE_FEATURES_PATH, HISTORICAL_FEATURES_PATH
from src.utils import ensure_directory

def main() -> None:
    ensure_directory(FEATURES_DIR)

    print("Golf Oracle Feature Pipeline")
    print(f"Features directory ready: {FEATURES_DIR}")
    print(f"Historical features path: {HISTORICAL_FEATURES_PATH}")
    print(f"Live features path: {LIVE_FEATURES_PATH}")
    print("Feature pipeline skeleton is working.")


if __name__ == "__main__":
    main()