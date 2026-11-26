import argparse
import json
from pathlib import Path

from ssnet_final.config import config_from_json_dict
from ssnet_final.engine.runner import run_train


def main():
    parser = argparse.ArgumentParser(description="Medical Image Segmentation Studio training worker")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8"))
    config = config_from_json_dict(values)
    run_train(config)


if __name__ == "__main__":
    main()
