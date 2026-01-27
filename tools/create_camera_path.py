import json
import math
import argparse
from pathlib import Path


def to4x4(c2w):
    if len(c2w) == 3:
        c2w += [[0, 0, 0, 1]]
    return c2w


def create_camera_path_json(transforms_json: Path, transforms_train_json: Path, output_filename: Path) -> None:
    transforms_data = json.loads(open(transforms_json).read())
    transforms_train_data = json.loads(open(transforms_train_json).read())

    output_filename.parent.mkdir(exist_ok=True)

    w, h = transforms_data["w"], transforms_data["h"]
    fl_y = transforms_data["fl_y"]
    fovy = (2 * math.atan(h * 0.5 / fl_y)) * 180 / math.pi

    new_transforms_data = {
        "camera_type": "perspective",
        "render_height": h,
        "render_width": w,
        "seconds": len(transforms_train_data),
        "camera_path": [
            {
                "camera_to_world": to4x4(frame["transform"]),
                "fov": fovy,
                "aspect": 1,
                "file_path": frame["file_path"],
            }
            for frame in transforms_train_data
        ],
    }

    with open(output_filename, mode="w") as f:
        f.write(json.dumps(new_transforms_data, indent=4))


def parse_args():
    parser = argparse.ArgumentParser(description="create camera_path.json")
    parser.add_argument("transforms_json", type=Path, help="from dataset")
    parser.add_argument("transforms_train_json", type=Path, help="from ns-export")
    parser.add_argument("--output-dir", type=Path, default=".")
    parser.add_argument("--output-filename", type=Path, default="camera_path.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_camera_path_json(args.transforms_json, args.transforms_train_json, args.output_dir / args.output_filename)
