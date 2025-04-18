import pickle
from pathlib import Path
import yaml
import numpy as np
import pandas as pd


def normalize_coords(x, y, img_w, img_h):
    return x / img_w, y / img_h


def convert_folder(folder: Path, keypoints, precision=6, verbose=False):
    # Find the pickle
    pickle_files = list(folder.glob("*.pickle"))
    if not pickle_files:
        print(f"[!] No .pickle file in {folder}")
        return

    # Load dataframe from pickle
    df = pd.read_pickle(pickle_files[0])
    scorer = df.columns.get_level_values(0)[0]
    bodyparts = df.columns.get_level_values(1).unique().tolist()

    # Loop through frames
    for idx, row in df.iterrows():
        for idx, row in df.iterrows():
            frame_name = str(idx)  # now correctly resolves to e.g. "BrownHorseinShadow_labeled_00001.png"


        img_file = folder / frame_name

        txt_file = img_file.with_suffix(".txt")

        # Get image size (you can hardcode if known)
        if not img_file.exists():
            print(f"[!] Image file {img_file} not found. Skipping.")
            continue
        img = img_file
        img_w, img_h = 1920, 1080  # Set your known image size here or detect it with OpenCV

        kp_list = []
        for part in keypoints:
            x = row[(scorer, part, "x")]
            y = row[(scorer, part, "y")]
            p = row[(scorer, part, "likelihood")]
            if np.isnan(x) or np.isnan(y) or p < 0.05:
                kp_list.append((0.0, 0.0, 0))
            else:
                nx, ny = normalize_coords(x, y, img_w, img_h)
                kp_list.append((nx, ny, 1))

        # Bounding box from visible keypoints
        vis = [(x, y) for x, y, v in kp_list if v == 1]
        if not vis:
            continue
        xs, ys = zip(*vis)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        w = max_x - min_x
        h = max_y - min_y

        # YOLO format: class_id cx cy w h x1 y1 v1 x2 y2 v2 ...
        with open(txt_file, "w") as f:
            f.write(f"0 {cx:.{precision}f} {cy:.{precision}f} {w:.{precision}f} {h:.{precision}f} ")
            f.write(" ".join(f"{x:.{precision}f} {y:.{precision}f} {v}" for x, y, v in kp_list))


def convert_dlc_to_yolo(
    dataset_path,
    train_paths,
    val_paths,
    keypoints,
    symmetric_pairs=None,
    data_yml_path=None,
    precision=6,
    verbose=False,
):
    dataset_path = Path(dataset_path)
    train_paths = [Path(p) for p in train_paths]
    val_paths = [Path(p) for p in val_paths]

    print("[*] Starting conversion...")

    for folder in train_paths + val_paths:
        if verbose:
            print(f"[+] Processing {folder.name}")
        convert_folder(folder, keypoints, precision=precision, verbose=verbose)

    if data_yml_path:
        flip_idx = list(range(len(keypoints)))
        if symmetric_pairs:
            for a, b in symmetric_pairs:
                flip_idx[a], flip_idx[b] = flip_idx[b], flip_idx[a]

        data = {
            "path": str(dataset_path.resolve()),
            "train": [str(p) for p in train_paths],
            "val": [str(p) for p in val_paths],
            "nc": 1,
            "names": {0: "horse"},
            "kpt_shape": [len(keypoints), 3],
            "flip_idx": flip_idx,
        }
        with open(data_yml_path, "w") as f:
            yaml.dump(data, f)
        print(f"[✓] Created {data_yml_path}")

    print("[✓] Conversion complete.")
