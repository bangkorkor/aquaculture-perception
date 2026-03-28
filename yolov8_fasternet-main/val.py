import argparse
import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("true", "1", "yes", "y", "on"):
        return True
    if v in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weight", type=str, required=True, help="path to trained model, e.g. best.pt")
    parser.add_argument("--data", type=str, required=True, help="data yaml path")

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)

    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")

    parser.add_argument("--plots", type=str2bool, default=False)
    parser.add_argument("--project", type=str, default="outputs/evaluation/RUOD")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--exist_ok", type=str2bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()

    model = YOLO(opt.weight)
    results = model.val(
        data=opt.data,
        split=opt.split,
        imgsz=opt.imgsz,
        batch=opt.batch,
        iou=opt.iou,
        conf=opt.conf,
        plots=opt.plots,
        project=opt.project,
        name=opt.name,
        exist_ok=opt.exist_ok,
    )


