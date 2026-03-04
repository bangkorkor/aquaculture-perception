from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader

TOPIC = "/sensor/sonoptix_echo/image"
MSGTYPE = "sensors/msg/SonoptixECHO"

def _normalize_tuple(tup):
    # rosbags may yield (t, raw, conn) or (conn, t, raw)
    t = next(x for x in tup if isinstance(x, (int, np.integer)))
    raw = next(x for x in tup if isinstance(x, (bytes, bytearray)))
    conn = next(x for x in tup if hasattr(x, "topic"))
    return int(t), raw, conn

def load_sonoptix_frame_from_bag(bag_path: Path, index: int) -> np.ndarray:
    """
    Returns float32 array shaped (H,W) for one SonoptixECHO frame.
    """
    assert bag_path.exists(), f"Bag not found: {bag_path}"
    with AnyReader([bag_path]) as r:
        conns = [c for c in r.connections if c.topic == TOPIC or c.msgtype == MSGTYPE]
        if not conns:
            raise RuntimeError(f"Topic {TOPIC} not found. Available:\n" + "\n".join(sorted({c.topic for c in r.connections})))
        conn = conns[0]

        for i, tup in enumerate(r.messages([conn])):
            if i == index:
                t_ns, raw, conn2 = _normalize_tuple(tup)
                msg = r.deserialize(raw, conn2.msgtype)
                data = np.asarray(msg.array_data.data, dtype=np.float32)

                dims = getattr(msg.array_data.layout, "dim", [])
                H = int(dims[0].size) if len(dims) > 0 else None
                W = int(dims[1].size) if len(dims) > 1 else None

                if H and W and H*W == data.size:
                    img = data.reshape(H, W)
                elif data.size == 1024*256:
                    img = data.reshape(1024, 256)
                elif data.size == 256*1024:
                    img = data.reshape(256, 1024)
                else:
                    for rows in (1024, 640, 512, 256, 128):
                        if data.size % rows == 0:
                            img = data.reshape(rows, data.size//rows)
                            break
                    else:
                        img = data.reshape(1, -1)

                return img, int(t_ns)

        raise IndexError(f"No message at index {index} on {TOPIC}")