import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from ultralytics import YOLO


if __name__ == "__main__":
    # Repo root = folder where this script lives
    ROOT = Path(__file__).resolve().parent

    # Adjust only these two if your run name/location is different
    weights_path = ROOT / "runsD" / "SDC" / "yolov11-sdc-uatd_120e_batch64_fair2" / "weights" / "best.pt"
    data_yaml = ROOT / "ultralytics" / "cfg" / "datasets" / "UATD.yaml"

    model = YOLO(str(weights_path))

    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=1,
        iou=0.45,
        conf=0.25,
        plots=False,
        project=str(ROOT / "outputsAAA" / "evaluation" / "UATD"),
        name="EVAL_yolov11s_uatd_120e_fair",
        exist_ok=True,
    )

    print(results)