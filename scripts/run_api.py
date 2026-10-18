"""Start the API with the host/port from configs/config.yaml.   python scripts/run_api.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402

from src.config import get_config  # noqa: E402

if __name__ == "__main__":
    c = get_config()["api"]
    uvicorn.run("api.main:app", host=c["host"], port=int(c["port"]), log_level="info")
