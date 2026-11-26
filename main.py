import os
from pathlib import Path

from ssnet_final.config import build_config
from ssnet_final.engine.runner import run_test, run_train


MODE = "train"
DATASET = "LA"  # this integrated project is 3-D only

# --------------------------- independent module choices ---------------------------
MODEL = "vnet"                  # vnet / catnet3d
FEATURE_LEARNING = "fsl"        # none / fsl
SEMI_SUPERVISION = "traditional"  # none / traditional / dnp
# ---------------------------------------------------------------------------------

LABELED_PERCENT = 10             # 5 / 10 / 100; semi-supervision uses 5 or 10
ANCHOR_TEMPERATURE = 100.0       # used only by catnet3d
DEVICE = "cuda:0"
SEED = 1337
WORKERS = 4
VALIDATION_INTERVAL = 200
SAVE_INTERVAL = 1000
SAVE_PREDICTIONS = True
LA_DATA_ROOT = os.environ.get("MEDICAL_SEG_DATA_ROOT", "").strip()
RUNS_ROOT = Path("runs")
RESUME_CHECKPOINT = None
TEST_CHECKPOINT = None


def main():
    if not LA_DATA_ROOT:
        raise SystemExit("请先设置 MEDICAL_SEG_DATA_ROOT 为 LA 数据目录，或通过 Streamlit 网页选择数据目录。")
    dataset = DATASET.upper()
    output_name = (
        f"{MODEL.lower()}_feat-{FEATURE_LEARNING.lower()}_"
        f"semi-{SEMI_SUPERVISION.lower()}_{dataset.lower()}_{LABELED_PERCENT}pct"
    )
    output = RUNS_ROOT / output_name
    resume = Path(RESUME_CHECKPOINT) if RESUME_CHECKPOINT is not None else None
    checkpoint = Path(TEST_CHECKPOINT) if TEST_CHECKPOINT is not None else output / "best.pt"

    config = build_config(
        command=MODE,
        dataset=dataset,
        model=MODEL,
        feature_learning=FEATURE_LEARNING,
        semi_supervision=SEMI_SUPERVISION,
        labeled_percent=LABELED_PERCENT,
        data_root=LA_DATA_ROOT,
        output=output,
        device=DEVICE,
        seed=SEED,
        workers=WORKERS,
        validation_interval=VALIDATION_INTERVAL,
        save_interval=SAVE_INTERVAL,
        resume=resume,
        checkpoint=checkpoint,
        save_predictions=SAVE_PREDICTIONS,
        anchor_temperature=ANCHOR_TEMPERATURE,
    )
    if config.command == "train":
        run_train(config)
    else:
        run_test(config)


if __name__ == "__main__":
    main()
