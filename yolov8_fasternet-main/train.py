import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"

from ultralytics import YOLO


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


TRAIN_PRESETS = {
    "ruod_120e_batch32_fair": {
        "epochs": 120,
        "imgsz": 640,
        "batch": 64,
        "nbs": 64,

        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,

        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "cos_lr": False,

        "amp": False,
        "device": "0",
        "workers": 4,
        "seed": 0,
        "deterministic": False,
        "pretrained": True,

        "fliplr": 0.5,
        "flipud": 0.0,

        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,

        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,

        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,

        "rect": False,
    }
}


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

    parser.add_argument(
        "--preset",
        type=str,
        default="ruod_120e_batch32_fair",
        choices=list(TRAIN_PRESETS.keys()) + ["none"],
        help="training preset name",
    )

    parser.add_argument("--yaml", type=str, default="ultralytics/models/v8/yolov8n.yaml", help="model.yaml path")
    parser.add_argument("--weight", type=str, default="", help="pretrained model path")
    parser.add_argument("--cfg", type=str, default="", help="extra cfg path if needed")
    parser.add_argument("--data", type=str, default="ultralytics/datasets/coco128.yaml", help="data yaml path")

    parser.add_argument("--epochs", type=int, default=None, help="number of epochs")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience")
    parser.add_argument("--batch", type=int, default=None, help="batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="input image size")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", default=None, help="cache ram/disk")
    parser.add_argument("--device", type=str, default=None, help="cuda device, e.g. 0 or 0,1 or cpu")
    parser.add_argument("--workers", type=int, default=None, help="dataloader workers")
    parser.add_argument("--project", type=str, default=str(ROOT / "runs/train"), help="save to project/name")
    parser.add_argument("--name", type=str, default="exp", help="run name")
    parser.add_argument("--resume", type=str, default=None, help="resume from checkpoint path")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["SGD", "Adam", "Adamax", "NAdam", "RAdam", "AdamW", "RMSProp", "auto"],
        default=None,
        help="optimizer",
    )
    parser.add_argument("--close_mosaic", type=int, default=None, help="disable mosaic for final epochs")
    parser.add_argument("--info", action="store_true", help="show model info/profile only")

    parser.add_argument("--save", type=str2bool, default=None, help="save checkpoints/results")
    parser.add_argument("--save-period", dest="save_period", type=int, default=-1, help="save checkpoint every x epochs")
    parser.add_argument("--exist-ok", dest="exist_ok", action="store_true", help="allow existing project/name")
    parser.add_argument("--seed", type=int, default=None, help="global seed")
    parser.add_argument("--deterministic", type=str2bool, default=None, help="deterministic mode")
    parser.add_argument("--single-cls", dest="single_cls", action="store_true", help="train as single class")
    parser.add_argument("--rect", type=str2bool, default=None, help="rectangular training")
    parser.add_argument("--cos-lr", dest="cos_lr", type=str2bool, default=None, help="cosine LR scheduler")
    parser.add_argument("--fraction", type=float, default=None, help="dataset fraction")
    parser.add_argument("--profile", action="store_true", help="profile ONNX/TRT during training")
    parser.add_argument("--pretrained", type=str2bool, default=None, help="use pretrained initialization")
    parser.add_argument("--amp", type=str2bool, default=None, help="use AMP")
    parser.add_argument("--unamp", action="store_true", help="force AMP off")

    parser.add_argument("--nbs", type=int, default=None, help="nominal batch size")
    parser.add_argument("--lr0", type=float, default=None, help="initial learning rate")
    parser.add_argument("--lrf", type=float, default=None, help="final LR factor")
    parser.add_argument("--momentum", type=float, default=None, help="optimizer momentum")
    parser.add_argument("--weight_decay", type=float, default=None, help="weight decay")
    parser.add_argument("--warmup_epochs", type=float, default=None, help="warmup epochs")
    parser.add_argument("--warmup_momentum", type=float, default=None, help="warmup momentum")

    parser.add_argument("--fliplr", type=float, default=None, help="left-right flip probability")
    parser.add_argument("--flipud", type=float, default=None, help="up-down flip probability")
    parser.add_argument("--degrees", type=float, default=None, help="rotation degrees")
    parser.add_argument("--translate", type=float, default=None, help="translation fraction")
    parser.add_argument("--scale", type=float, default=None, help="scale gain")
    parser.add_argument("--shear", type=float, default=None, help="shear degrees")
    parser.add_argument("--perspective", type=float, default=None, help="perspective factor")

    parser.add_argument("--hsv_h", type=float, default=None, help="HSV-H augmentation")
    parser.add_argument("--hsv_s", type=float, default=None, help="HSV-S augmentation")
    parser.add_argument("--hsv_v", type=float, default=None, help="HSV-V augmentation")

    parser.add_argument("--mosaic", type=float, default=None, help="mosaic probability")
    parser.add_argument("--mixup", type=float, default=None, help="mixup probability")
    parser.add_argument("--copy_paste", type=float, default=None, help="copy-paste probability")

    parser.add_argument("--overlap_mask", type=str2bool, default=None, help="segment only")
    parser.add_argument("--mask_ratio", type=int, default=None, help="segment only")
    parser.add_argument("--dropout", type=float, default=None, help="classify only")

    return parser.parse_known_args()[0]


def build_train_args(opt):
    cli = vars(opt).copy()

    preset_name = cli.pop("preset", "none")
    cli.pop("yaml", None)
    cli.pop("weight", None)
    cli.pop("info", None)
    cli.pop("cfg", None)

    unamp = cli.pop("unamp", False)
    if unamp:
        cli["amp"] = False

    train_args = {}
    if preset_name != "none":
        train_args.update(TRAIN_PRESETS[preset_name])

    for k, v in cli.items():
        if v is not None and v != "":
            train_args[k] = v

    return train_args


class YOLOV8(YOLO):
    def __init__(self, yaml="ultralytics/models/v8/yolov8n.yaml", weight="", task=None):
        super().__init__(yaml, task)
        if weight:
            self.load(weight)


if __name__ == "__main__":
    opt = parse_opt()
    model = YOLOV8(yaml=opt.yaml, weight=opt.weight)

    if opt.info:
        imgsz = opt.imgsz if opt.imgsz is not None else TRAIN_PRESETS["ruod_120e_batch32_fair"]["imgsz"]
        model.info(detailed=True, verbose=True)
        model.profile(imgsz)
        print("before fuse...")
        model.info(detailed=False, verbose=True)
        print("after fuse...")
        model.fuse()
    else:
        train_args = build_train_args(opt)
        print("Training args:")
        for k in sorted(train_args):
            print(f"  {k}: {train_args[k]}")
        model.train(**train_args)