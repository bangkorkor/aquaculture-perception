#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader

# --- CONFIG ---
BAG = Path("../../../uw_yolov8/data/SOLAQUA/2024-08-20_13-57-42_video.bag")
TOPIC = "/sensor/sonoptix_echo/image"
FRAME_IDX = 1000
H_FOV_DEG = 90.0
MAX_RANGE_M = 5.0
TVG_DB_PER_M = 0.0
ROTATE = "none"   # options: "none", "ccw90", "cw90", "rot180"
# --------------

def first_hw(layout):
    h = w = None
    for d in layout.dim:
        if d.label == "height" and h is None:
            h = int(d.size)
        if d.label == "width" and w is None:
            w = int(d.size)
        if h and w:
            break
    return h or 1024, w or 256

def lognorm(a):
    a = a.astype(np.float32)
    a = np.log1p(a - a.min())
    a /= (a.max() if a.max() > 0 else 1.0)
    return a

def apply_rotation(img, how):
    if how == "none":
        return img, "no rotation"
    if how == "ccw90":
        return np.rot90(img, k=1), "rotated 90° CCW"
    if how == "cw90":
        return np.rot90(img, k=3), "rotated 90° CW"
    if how == "rot180":
        return np.rot90(img, k=2), "rotated 180°"
    raise ValueError(f"Unknown ROTATE='{how}'")

def main():
    print("\n=== DEBUG START ===")
    print(f"[DEBUG] BAG path: {BAG}  (exists={BAG.exists()})")
    if not BAG.exists():
        print("[ERROR] Bag path is wrong.")
        return

    with AnyReader([BAG]) as reader:
        print("[DEBUG] Opened bag successfully.")
        conns = [c for c in reader.connections if c.topic == TOPIC]
        print(f"[DEBUG] Found {len(conns)} connections for topic '{TOPIC}'")
        if not conns:
            print("[ERROR] Topic not found. Available:", [c.topic for c in reader.connections])
            return
        conn = conns[0]

        # count frames
        frame_count = sum(1 for _ in reader.messages(connections=[conn]))
        print(f"[DEBUG] Total frames available: {frame_count}")
        if FRAME_IDX >= frame_count:
            print(f"[ERROR] Requested frame {FRAME_IDX}, but only {frame_count} frames exist.")
            return

        # read requested frame
        print(f"[DEBUG] Reading frame #{FRAME_IDX} ...")
        img = None
        for idx, (_, ts, raw) in enumerate(reader.messages(connections=[conn])):
            if idx == FRAME_IDX:
                print(f"[DEBUG] -> Found frame {idx}, timestamp={ts}")
                msg = reader.deserialize(raw, conn.msgtype)
                fa = msg.array_data
                h, w = first_hw(fa.layout)
                data = np.asarray(fa.data, dtype=np.float32)
                print(f"[DEBUG] Layout: H={h}, W={w}, data_len={len(data)}")
                if len(data) < h * w:
                    print("[ERROR] Not enough data to reshape into image.")
                    return
                data = data[:h*w].reshape(h, w)

                if TVG_DB_PER_M != 0:
                    r = np.linspace(0, MAX_RANGE_M, h, dtype=np.float32)
                    gain = 10 ** ((TVG_DB_PER_M * r) / 20.0)
                    data *= gain[:, None]
                    print("[DEBUG] Applied TVG")

                img = lognorm(data)
                print("[DEBUG] Applied log-normalization; img shape:", img.shape)
                break

    if img is None:
        print("[ERROR] No image decoded.")
        return

    # apply requested rotation
    img_rot, rot_msg = apply_rotation(img, ROTATE)
    print(f"[DEBUG] Orientation: {rot_msg}; new shape={img_rot.shape}")

    # axes (keep the same physical axes; pixel count may change, that’s fine)
    x_deg = np.linspace(-H_FOV_DEG/2, H_FOV_DEG/2, img_rot.shape[1])
    y_m   = np.linspace(0, MAX_RANGE_M,      img_rot.shape[0])
    extent = [x_deg[0], x_deg[-1], y_m[0], y_m[-1]]
    print(f"[DEBUG] x-axis: {x_deg[0]}..{x_deg[-1]} deg, y-axis: {y_m[0]}..{y_m[-1]} m")

    # plot + save
    print("[DEBUG] Plotting and saving...")
    plt.figure(figsize=(8,8))
    hnd = plt.imshow(img_rot, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("Beam angle [deg]")
    plt.ylabel("Range [m]")
    plt.title(f"Raw (frame {FRAME_IDX}) — {rot_msg}")
    plt.colorbar(hnd, label="normalized intensity")
    plt.tight_layout()
    out = f"echo_frame_{FRAME_IDX}_{ROTATE}.png"
    plt.savefig(out, dpi=200)
    print(f"[DEBUG] Saved plot to: {out}")
    plt.show()
    print("\n=== DEBUG END ===")

if __name__ == "__main__":
    main()
