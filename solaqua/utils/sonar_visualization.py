# sonar_visualization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from typing import Union, Optional, Callable
from pathlib import Path
from datetime import datetime, timezone
import re
import subprocess
from typing import Tuple
import cv2



# ================ default intensity enhancer ====================
def _tvg_amplitude_curve(H: int, rmin: float, rmax: float, alpha_db_per_m: float, r0: float) -> np.ndarray:
    r = np.linspace(rmin, rmax, H, dtype=np.float32)
    if alpha_db_per_m <= 0:
        return np.ones_like(r, dtype=np.float32)
    k = alpha_db_per_m / 8.685889638  # 20*log10(e)
    return np.exp(k * np.maximum(0.0, r - r0)).astype(np.float32)

def enhance_intensity(
    M: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    Z = np.asarray(M, dtype=np.float32)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    # TVG
    if str(cfg["enh_tvg"]).lower() == "amplitude":
        G = _tvg_amplitude_curve(Z.shape[0], cfg["range_min_m"], cfg["range_max_m"],
                                 cfg["enh_alpha_db_per_m"], cfg["enh_r0"])[:, None]
        Z = Z * G

    # scaling
    s = str(cfg["enh_scale"]).lower()
    eps = float(cfg["enh_eps_log"])
    if s == "db":
        Zs = 20.0 * np.log10(Z + eps)
    elif s == "log":
        Zs = np.log10(Z + eps)
    elif s == "sqrt":
        Zs = np.sqrt(Z)
    else:
        Zs = Z

    # percentile normalization
    lo, hi = np.percentile(Zs, [cfg["enh_p_low"], cfg["enh_p_high"]])
    if hi <= lo:
        hi = lo + 1e-6
    Zs = (np.clip(Zs, lo, hi) - lo) / (hi - lo)

    if cfg["enh_zero_aware"]:
        Zs[Z <= eps] = 0.0

    g = float(cfg["enh_gamma"])
    if g != 1.0:
        Zs = np.clip(Zs, 0.0, 1.0) ** g

    return np.clip(Zs, 0.0, 1.0).astype(np.float32)

# ========== custom enhancer to match cfc sonar dataset =================
def apply_gaussian_noise_01(Z: np.ndarray, cfg: dict, *, seed=None) -> np.ndarray:
    """
    Add simple additive Gaussian noise to a [0,1] image Z.
    Config:
        ds_noise_std: float   (e.g., 0.02)
        ds_noise_seed: int or None
        ds_noise_enabled: bool (optional)
    """
    if not cfg.get("ds_noise_enabled", False):
        return Z

    std = float(cfg.get("ds_noise_std", 0.02))  # default: 2% noise
    if std <= 0:
        return Z

    # Random generator (optional reproducibility)
    rnd = np.random.RandomState(cfg.get("ds_noise_seed") if seed is None else seed)

    noise = rnd.normal(0.0, std, size=Z.shape).astype(np.float32)
    Zn = Z + noise

    return np.clip(Zn, 0.0, 1.0).astype(np.float32)

def _range_axis(H, rmin, rmax, dtype=np.float32):
    # inclusive end points across H bins
    return np.linspace(float(rmin), float(rmax), int(H), dtype=dtype)

def enhance_cfc_style(M: np.ndarray, cfg: dict) -> np.ndarray:
    """ enhance_cfc_style
    Produce a [0,1] image styled like the reference dataset:
      - more uniform brightness over range
      - bright targets pop (white)
      - background pushed toward mid-gray (but controllable)
      - robust to speckle via per-range baseline removal
    """
    # --- 0) sanitize ---
    Z = np.asarray(M, dtype=np.float32)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    H, W = Z.shape

    rmin = float(cfg["range_min_m"])
    rmax = float(cfg["range_max_m"])
    rng = _range_axis(H, rmin, rmax)[:, None]  # (H,1)


    # --- 2) TVG: geometric spreading + absorption ---
    # amplitude ~ 1/r^g  (g in [0, 2]); compensate by multiplying by r^g
    g_geo = float(cfg.get("ds_geo_exponent", 1.0))  # 1.0 is common for amplitude
    eps_r = 1e-6
    G_geo = np.power(np.maximum(rng, eps_r), g_geo, dtype=np.float32)  # (H,1)

    # absorption (dB per meter). Convert to natural log scale for amplitudes.
    alpha_db_per_m = float(cfg.get("ds_alpha_db_per_m", 0.0))
    if alpha_db_per_m > 0.0:
        k = alpha_db_per_m / 8.685889638  # 20*log10(e)
        r0 = float(cfg.get("ds_alpha_r0", 0.0))
        G_abs = np.exp(k * np.maximum(0.0, rng - r0)).astype(np.float32)
    else:
        G_abs = 1.0

    Z = Z * G_geo * G_abs  # (H,W)

    # --- 3) Per-range background flattening (remove water-column baseline) ---
    # subtract a percentile across beams for each row; keep positives
    p_bg = float(cfg.get("ds_bg_percentile", 60.0))  # 50..70 works well
    bg = np.percentile(Z, p_bg, axis=1, keepdims=True).astype(np.float32)
    scale_bg = float(cfg.get("ds_bg_scale", 1.0))    # 0.8..1.2
    Z = Z - scale_bg * bg
    Z = np.maximum(Z, 0.0, dtype=np.float32)

    # --- 4) Log/dB compression ---
    eps = float(cfg.get("ds_eps_log", 1e-5))
    Zc = 20.0 * np.log10(Z + eps)  # dB-like scale

    # --- 5) Robust contrast stretch to [0,1] ---
    p_low  = float(cfg.get("ds_p_low", 2.0))
    p_high = float(cfg.get("ds_p_high", 99.7))
    lo, hi = np.percentile(Zc, [p_low, p_high])
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    Zn = (np.clip(Zc, lo, hi) - lo) / (hi - lo)

    # --- 6) Gamma: lift highlights a bit ---
    gamma = float(cfg.get("ds_gamma", 0.9))  # <1 brightens mid/high
    if gamma != 1.0:
        Zn = np.clip(Zn, 0.0, 1.0) ** gamma

    
    # --- 6.5) (Optional) Noise: adds noise
    Zn = apply_gaussian_noise_01(Z, cfg)

    # --- 7) Soft floor to push background to mid-gray (optional) ---
    # The reference image background looks ~0.3–0.4. This keeps weak speckle visible.
    soft_floor = float(cfg.get("ds_soft_floor", 0.0))  # e.g. 0.15..0.35; 0 disables
    if soft_floor > 0.0:
        Zn = soft_floor + (1.0 - soft_floor) * np.clip(Zn, 0.0, 1.0)

    # --- 7.5) Hard white cap ---
    max_cap = float(cfg.get("ds_max_white_cap", 1.0))
    if max_cap < 1.0:
        Zn = np.minimum(Zn, max_cap)

    return np.clip(Zn, 0.0, 1).astype(np.float32)

# ========= binary enhanceer used to making labeling easy
def enhance_bw(M: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Multi-effect sonar enhancer combining:
      - denoise
      - cluster dilation boost
      - robust normalization
      - gamma curve
      - edgy highlight boost
      - (optional) unsharp mask sharpening
      - (optional) CLAHE local contrast
      - (optional) cluster-only brightness mask
      - (optional) highlight rim effect
    All effects are controlled by config parameters.
    """
    Z = np.asarray(M, dtype=np.float32)
    Z = np.nan_to_num(Z, nan=0.0, neginf=0.0, posinf=0.0)

    # ---------------------------------------------------------------
    # 1) Base denoise
    # ---------------------------------------------------------------
    k_denoise = int(cfg.get("denoise_kernel", 3))
    if k_denoise >= 3:
        Z = cv2.GaussianBlur(Z, (k_denoise, k_denoise), 0)

    # ---------------------------------------------------------------
    # 2) Cluster dilation boost
    # ---------------------------------------------------------------
    k_cluster = int(cfg.get("cluster_kernel", 5))
    if k_cluster > 1:
        Z_dil = cv2.dilate(Z, np.ones((k_cluster, k_cluster), np.uint8))
        w = float(cfg.get("cluster_blend", 0.4))
        Z = (1-w) * Z + w * Z_dil

    # ---------------------------------------------------------------
    # 3) Robust global normalization to [0,1]
    # ---------------------------------------------------------------
    p_low  = float(cfg.get("p_low", 5.0))
    p_high = float(cfg.get("p_high", 99.5))
    lo, hi = np.percentile(Z, [p_low, p_high])
    hi = max(hi, lo + 1e-6)
    Z = (np.clip(Z, lo, hi) - lo) / (hi - lo)

    # ---------------------------------------------------------------
    # 4) Gamma curve
    # ---------------------------------------------------------------
    gamma = float(cfg.get("gamma", 0.7))
    Z = np.clip(Z, 0, 1) ** gamma

    # ---------------------------------------------------------------
    # 5) Edgy highlight boost
    # ---------------------------------------------------------------
    edgy = float(cfg.get("edgy_boost", 0.25))
    if edgy > 0:
        Z = Z + edgy * (Z ** 2)
        Z = np.clip(Z, 0, 1)

    # ---------------------------------------------------------------
    # 6) OPTIONAL: Unsharp mask (sharpening)
    # ---------------------------------------------------------------
    if cfg.get("sharpen_enabled", False):
        amount = float(cfg.get("sharpen_amount", 1.0))  # 0.5–2.0
        blur = cv2.GaussianBlur(Z, (0,0), sigmaX=3)
        Z = np.clip(Z + amount * (Z - blur), 0, 1)

    # ---------------------------------------------------------------
    # 7) OPTIONAL: CLAHE – local contrast (strong)
    # ---------------------------------------------------------------
    if cfg.get("clahe_enabled", False):
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg.get("clahe_clip", 2.0)),
            tileGridSize=(8, 8)
        )
        Z8 = np.uint8(Z * 255)
        Z8 = clahe.apply(Z8)
        Z = Z8.astype(np.float32) / 255.0

    # ---------------------------------------------------------------
    # 8) OPTIONAL: Cluster-only brightness mask
    #     (boost only areas that are spatially connected)
    # ---------------------------------------------------------------
    if cfg.get("cluster_mask_enabled", False):
        thr = float(cfg.get("cluster_mask_thr", 0.3))
        mask = (Z > thr).astype(np.float32)
        mask = cv2.dilate(mask, np.ones((5,5), np.uint8))
        strength = float(cfg.get("cluster_mask_strength", 0.4))
        Z = Z + strength * Z * mask
        Z = np.clip(Z, 0, 1)

    # ---------------------------------------------------------------
    # 9) OPTIONAL: Highlight rim / glow
    # ---------------------------------------------------------------
    if cfg.get("rim_enabled", False):
        # detect edges
        Z8 = np.uint8(Z * 255)
        edges = cv2.Canny(Z8, 30, 120)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8))
        edges = cv2.GaussianBlur(edges.astype(np.float32), (5,5), 0) / 255.0

        rim_strength = float(cfg.get("rim_strength", 0.2))  # 0.1–0.4
        Z = np.clip(Z + rim_strength * edges, 0, 1)

    return Z.astype(np.float32)



# ==== raw plot =======
def plot_raw_frame(M: np.ndarray, frame_index: int, cfg: dict):
    """
    Plot the raw sonar frame 
    """
    # --- orientation operations directly ---
    Z = M.copy()
    if cfg.get("transpose_M", False): # returns cfg[key] if it exists, false otherwise
        Z = Z.T     # we transpose the matrix (swap H and W)
    if cfg.get("flipY_m", False):
        Z = Z[::-1, :]  # we flip the beam angles 
    if cfg.get("flipX_m", False):
        Z = Z[:, ::-1]  # we flip the range angles

    # --- imshow extent in angle+range coordinates ---
    theta_min = -0.5 * float(cfg["fov_deg"])
    theta_max = +0.5 * float(cfg["fov_deg"])
    extent = (
        theta_min, theta_max,
        float(cfg["range_min_m"]), float(cfg["range_max_m"])
    )

    fig, ax = plt.subplots(
        figsize=cfg.get("figsize", (6.0, 5.6)),
        constrained_layout=True
    )

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cfg.get("cmap_raw", "viridis")
    )

    ax.set_title(f"Raw (frame {frame_index})")
    ax.set_xlabel("Beam angle [deg]")
    ax.set_ylabel("Range [m]")

    # apply display crop
    ax.set_ylim(cfg["display_range_min_m"], cfg["display_range_max_m"])

    fig.colorbar(im, ax=ax, label="Echo (raw units)")

    return fig


# --- enhanced plot ------------------------------------------------------------
def plot_enhanced_frame(M: np.ndarray, frame_index: int, cfg: dict, enhancer: callable = None):
    """
    Plot  the enhanced sonar frame (after orientation + intensity enhancement).
    """
    # --- orientation (same as raw) ---
    Z = M.copy()
    if cfg.get("transpose_M", False): # returns cfg[key] if it exists, false otherwise
        Z = Z.T     # we transpose the matrix (swap H and W)
    if cfg.get("flipY_m", False):
        Z = Z[::-1, :]  # we flip the beam angles 
    if cfg.get("flipX_m", False):
        Z = Z[:, ::-1]  # we flip the range angles

    
    
    # --- choose enhancer ---
    # Default if none provided → enhance_intensity
    if enhancer is None:
        enhancer = enhance_intensity

    # Apply enhancer
    Z_enh = enhancer(Z, cfg)


    
    # --- imshow extent ---
    theta_min = -0.5 * float(cfg["fov_deg"])
    theta_max = +0.5 * float(cfg["fov_deg"])
    extent = (
        theta_min, theta_max,
        float(cfg["range_min_m"]), float(cfg["range_max_m"])
    )

    fig, ax = plt.subplots(
        figsize=cfg.get("figsize", (6.0, 5.6)),
        constrained_layout=True
    )

    im = ax.imshow(
        Z_enh,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=0.0, vmax=1.0,
        cmap=cfg.get("cmap_enh", "magma")
    )

    ax.set_title(f"Enhanced (frame {frame_index})")
    ax.set_xlabel("Beam angle [deg]")
    ax.set_ylabel("Range [m]")

    # apply display crop
    ax.set_ylim(cfg["display_range_min_m"], cfg["display_range_max_m"])


    fig.colorbar(im, ax=ax, label="Echo (enhanced)")

    return fig




# --- rasterizer with centered (symmetric) view around the mid-angle ---
def cone_rasterizer_display_cell(
    Z: np.ndarray,              # (H, W) = (range bins, beams)
    fov_deg: float,
    range_min_m: float,         # physical mapping
    range_max_m: float,         # physical mapping
    coneview_range_min_m: float,
    coneview_range_max_m: float,
    coneview_angle_min_deg: float,
    coneview_angle_max_deg: float,
    img_w: int,
    img_h: int,
    *,
    rotate_deg: float = 0.0,    # additional rotation (deg)
    bg_value: float = np.nan,
):
    """
    Rasterize Z into a Cartesian fan, honoring a user display window:
      - range:  [coneview_range_min_m, coneview_range_max_m]
      - angles: [coneview_angle_min_deg, coneview_angle_max_deg]
    The output grid is **symmetric** about the mid-angle of [amin, amax].
    """
    H, W = Z.shape
    fov_half = 0.5 * float(fov_deg)

    # ---- VALIDATION ----
    r_phys_min = float(range_min_m)
    r_phys_max = float(range_max_m)
    r0 = float(coneview_range_min_m)
    r1 = float(coneview_range_max_m)
    amin = float(coneview_angle_min_deg)
    amax = float(coneview_angle_max_deg)

    if not (r_phys_min <= r0 < r1 <= r_phys_max):
        raise ValueError(
            f"coneview_range_[min,max]_m must satisfy {r_phys_min} <= min < max <= {r_phys_max} "
            f"(got min={r0}, max={r1})"
        )
    if amin >= amax:
        raise ValueError(f"coneview_angle_min_deg must be < coneview_angle_max_deg (got {amin} >= {amax})")
    if not (-fov_half <= amin and amax <= fov_half):
        raise ValueError(
            f"angles must lie within physical FOV [{-fov_half}, {+fov_half}] deg "
            f"(got [{amin}, {amax}])"
        )

    # Mid-angle centering
    amid  = 0.5 * (amin + amax)            # center of the window
    ahalf = 0.5 * (amax - amin)            # half-width of the window (>=0)

    # ---- OUTPUT GRID (centered around amid) ----
    # Build a symmetric X-range around the centerline so the view is visually centered.
    y_min, y_max = 0.0, r1
    x_span = y_max * np.sin(np.deg2rad(ahalf))  # symmetric half-width at the far radius
    x_min, x_max = -x_span, +x_span

    ys = np.linspace(y_min, y_max, img_h, endpoint=False, dtype=np.float32)
    xs = np.linspace(x_min, x_max, img_w, endpoint=False, dtype=np.float32)
    Xc, Yc = np.meshgrid(xs, ys)  # (img_h, img_w)

    # Polar of output pixels in the **centered view frame**:
    # theta_rel = 0 is along the mid-angle ray.
    theta_rel = np.rad2deg(np.arctan2(Xc, Yc))   # [-180,180], right is +, left is -
    rng       = np.hypot(Xc, Yc)

    # Effective beam angle in the sonar frame that we must sample:
    # add back the mid-angle, then apply user rotation.
    theta_eff = theta_rel + amid - float(rotate_deg)

    # ---- MAP to Z indices ----
    # Beams uniformly span [-fov/2, +fov/2] in the sonar frame
    th_min, th_max = -fov_half, +fov_half
    beam_idx = (theta_eff - th_min) / (th_max - th_min) * (W - 1)
    beam_idx = np.clip(beam_idx, 0.0, W - 1)

    # Ranges uniformly span [range_min_m, range_max_m]
    r_idx = (rng - r_phys_min) / (r_phys_max - r_phys_min) * (H - 1)
    r_idx = np.clip(r_idx, 0.0, H - 1)

    # ---- MASK: keep only the requested window and physical bounds ----
    mask = (
        (theta_rel < -ahalf) | (theta_rel > +ahalf) |  # angle window in centered frame
        (rng < r0) | (rng > r1) |                      # range window
        (theta_eff < th_min) | (theta_eff > th_max)    # safety vs physical FOV
    )

    # ---- BILINEAR SAMPLE ----
    r0i = np.floor(r_idx).astype(np.int32)
    c0i = np.floor(beam_idx).astype(np.int32)
    r1i = np.clip(r0i + 1, 0, H - 1)
    c1i = np.clip(c0i + 1, 0, W - 1)

    fr = (r_idx - r0i).astype(np.float32)
    fc = (beam_idx - c0i).astype(np.float32)

    Z00 = Z[r0i, c0i]; Z10 = Z[r1i, c0i]
    Z01 = Z[r0i, c1i]; Z11 = Z[r1i, c1i]
    top = Z00 * (1.0 - fc) + Z01 * fc
    bot = Z10 * (1.0 - fc) + Z11 * fc
    Zi  = top * (1.0 - fr) + bot * fr

    Zi = Zi.astype(np.float32)
    Zi[mask] = bg_value
    return Zi, (float(x_min), float(x_max), float(y_min), float(y_max)), amid, ahalf


# --- plotting wrapper (reads coneview_*; centered display) ---
def plot_cone_view(
    M: np.ndarray,
    cfg: dict,
    *,
    use_enhanced: bool = True,
    enhancer: callable = None,
):
    # Orientation
    Z = M
    if cfg.get("transpose_M", False):
        Z = Z.T
    if cfg.get("flipY_m", False):
        Z = Z[::-1, :]
    if cfg.get("flipX_m", False):
        Z = Z[:, ::-1]

    # Optional enhancement
    if use_enhanced and enhancer is not None:
        Z = enhancer(Z, cfg)

    # Physical limits
    fov = float(cfg["fov_deg"])
    r_phys_min = float(cfg["range_min_m"])
    r_phys_max = float(cfg["range_max_m"])
    a_full_min = -0.5 * fov
    a_full_max = +0.5 * fov

    # User window (defaults to full)
    cv_rmin = float(cfg.get("coneview_range_min_m", r_phys_min))
    cv_rmax = float(cfg.get("coneview_range_max_m", r_phys_max))
    cv_amin = float(cfg.get("coneview_angle_min_deg", a_full_min))
    cv_amax = float(cfg.get("coneview_angle_max_deg", a_full_max))

    # Rasterize (centered)
    cone, (x_min, x_max, y_min, y_max), amid, ahalf = cone_rasterizer_display_cell(
        Z,
        fov_deg=fov,
        range_min_m=r_phys_min,
        range_max_m=r_phys_max,
        coneview_range_min_m=cv_rmin,
        coneview_range_max_m=cv_rmax,
        coneview_angle_min_deg=cv_amin,
        coneview_angle_max_deg=cv_amax,
        img_w=int(cfg["img_w"]),
        img_h=int(cfg["img_h"]),
        rotate_deg=float(cfg.get("rotate_deg", 0.0)),
        bg_value=np.nan,
    )

    # Plot
    fig, ax = plt.subplots(figsize=(9.5, 8.0), constrained_layout=True)
    # fig.patch.set_facecolor("black"); ax.set_facecolor("black")

    cmap_name = cfg["cmap_enh" if use_enhanced else "cmap_raw"]
    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(cfg.get("bg_color", "#4b4b4b"))  # the gray should be #2596be (to match cfc)

    if use_enhanced:
        vmin, vmax = 0.0, 1.0
    else:
        with np.errstate(all='ignore'):
            vmin = float(np.nanmin(cone)); vmax = float(np.nanmax(cone))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0

    im = ax.imshow(
        cone,
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        aspect="equal",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
    )

    ax.set_xlabel("Starboard X [m] (+)")
    ax.set_ylabel("Forward Y [m]")
    ax.set_title("Sonar Cone — " + ("Enhanced" if use_enhanced else "Raw"))

    # Keep the radial extent of the chosen window
    ax.set_ylim(0.0, cv_rmax)


    fig.colorbar(
        im, ax=ax, pad=0.02, shrink=0.9,
        label=("Echo (normalized)" if use_enhanced else "Echo (raw units)")
    )
    return fig



# =========== Export scripts


from PIL import Image

def save_cone_view_image(
    M: np.ndarray,
    cfg: dict,
    out_path: Union[str, Path],
    *,
    use_enhanced: bool = True,
    enhancer: Optional[Callable[[np.ndarray, dict], np.ndarray]] = None,
    transparent_bg: bool = False,  # set True if you prefer transparent NaN background
):
    # --- Orientation (same as plot_cone_view) ---
    Z = M
    if cfg.get("transpose_M", False):
        Z = Z.T
    if cfg.get("flipY_m", False):
        Z = Z[::-1, :]
    if cfg.get("flipX_m", False):
        Z = Z[:, ::-1]

    if use_enhanced and enhancer is not None:
        Z = enhancer(Z, cfg)

    # --- Rasterize (same call/signature you use in plot_cone_view) ---
    cone, _, _, _ = cone_rasterizer_display_cell(
        Z,
        fov_deg=float(cfg["fov_deg"]),
        range_min_m=float(cfg["range_min_m"]),
        range_max_m=float(cfg["range_max_m"]),
        coneview_range_min_m=float(cfg.get("coneview_range_min_m", cfg["range_min_m"])),
        coneview_range_max_m=float(cfg.get("coneview_range_max_m", cfg["range_max_m"])),
        coneview_angle_min_deg=float(cfg.get("coneview_angle_min_deg", -0.5 * cfg["fov_deg"])),
        coneview_angle_max_deg=float(cfg.get("coneview_angle_max_deg", +0.5 * cfg["fov_deg"])),
        img_w=int(cfg["img_w"]),
        img_h=int(cfg["img_h"]),
        rotate_deg=float(cfg.get("rotate_deg", 0.0)),
        bg_value=np.nan,
    )

    # --- Match imshow(origin="lower"): flip vertically ---
    cone_disp = np.flipud(cone)

    # --- Match vmin/vmax logic from plot_cone_view ---
    if use_enhanced:
        vmin, vmax = 0.0, 1.0
    else:
        with np.errstate(all='ignore'):
            vmin = float(np.nanmin(cone_disp))
            vmax = float(np.nanmax(cone_disp))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0

    # Normalize to 0..1 for colormap
    with np.errstate(invalid='ignore', divide='ignore'):
        norm = (cone_disp - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    # Colormap and background handling (like cmap.set_bad)
    cmap_name = cfg["cmap_enh" if use_enhanced else "cmap_raw"]
    cmap = plt.cm.get_cmap(cmap_name)
    rgba = cmap(norm)  # shape (H,W,4)
    nan_mask = np.isnan(cone_disp)

    if transparent_bg:
        # Transparent where NaN
        rgba[nan_mask, 3] = 0.0
    else:
        # Opaque background color from cfg (default #4b4b4b)
        bg_hex = cfg.get("bg_color", "#4b4b4b").lstrip("#")
        bg_rgb = tuple(int(bg_hex[i:i+2], 16) for i in (0, 2, 4))
        rgba[nan_mask, :3] = np.array(bg_rgb) / 255.0
        rgba[nan_mask, 3] = 1.0

    # Convert to 8-bit and save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = (rgba[:, :, :3] * 255).astype(np.uint8) if not transparent_bg else (rgba * 255).astype(np.uint8)
    Image.fromarray(img, mode="RGB" if not transparent_bg else "RGBA").save(out_path)
    # print(f"✅ Saved cone-view image to {out_path}")


    

# ==== helper =========
def ns_to_utc(ns_timestamp: int) -> datetime:
    """Convert a nanosecond Unix timestamp to UTC datetime."""
    seconds = ns_timestamp / 1e9
    return datetime.fromtimestamp(seconds, tz=timezone.utc)




# ========== Make mp4 from frames folder, frames needs ns-timestamp for names 


def build_vfr_mp4_from_ns_frames(
    frames_dir: Path,
    out_mp4: Path,
    *,
    pattern: str = "*.jpg",
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    min_dur_s: float = 1/120,
    max_dur_s: float = 1.0,
    speed: float = 1.0,
) -> Path:
    frames_dir = Path(frames_dir)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    pat = re.compile(r"^(\d+)\.jpg$", re.IGNORECASE)
    files = sorted([p for p in frames_dir.glob(pattern) if pat.match(p.name)],
                   key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"No frames in {frames_dir} matching {pattern} with numeric names.")

    ts = [int(p.stem) for p in files]  # ns → s durations
    durs = []
    for i in range(len(ts) - 1):
        dt = (ts[i+1] - ts[i]) / 1e9
        dt = max(min_dur_s, min(max_dur_s, dt)) / max(1e-9, speed)
        durs.append(dt)
    if not durs:
        durs = [1.0 / max(1e-9, speed)]

    list_txt = out_mp4.with_suffix(".list.txt")
    with list_txt.open("w", encoding="utf-8") as f:
        for p, dur in zip(files[:-1], durs):
            abs_path = p.resolve()  # <-- absolute
            f.write(f"file '{abs_path.as_posix()}'\n")
            f.write(f"duration {dur:.9f}\n")
        # repeat last frame once (concat demuxer quirk)
        last_abs = files[-1].resolve()
        f.write(f"file '{last_abs.as_posix()}'\n")
        f.write(f"file '{last_abs.as_posix()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_txt),
        "-fps_mode", "vfr",              # modern flag
        "-pix_fmt", "yuv420p",
        "-c:v", codec, "-crf", str(crf), "-preset", preset,
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ Wrote {out_mp4}")

    # cleanup
    try:
        list_txt.unlink()
        print(f"🧹 Deleted temporary file: {list_txt.name}")
    except Exception as e:
        print(f"⚠️ Could not delete {list_txt}: {e}")

    return out_mp4





# ========= make joint vision and sonar video, correctly timed

def _collect_ts_sorted(folder: Path) -> Tuple[list[Path], list[int]]:
    pat = re.compile(r"^(\d+)\.jpg$", re.IGNORECASE)
    files = [p for p in folder.glob("*.jpg") if pat.match(p.name)]
    if not files:
        raise FileNotFoundError(f"No timestamp-named JPGs in {folder}")
    files.sort(key=lambda p: int(p.stem))
    ts = [int(p.stem) for p in files]  # ns
    return files, ts

def _write_concat_list(files, ts, list_path: Path, *, min_dur=1/120, max_dur=1.0, speed=1.0):
    list_path.parent.mkdir(parents=True, exist_ok=True)
    # per-frame durations from deltas (seconds)
    durs = []
    for i in range(len(ts)-1):
        dt = (ts[i+1] - ts[i]) / 1e9
        dt = max(min_dur, min(max_dur, dt)) / max(1e-9, speed)
        durs.append(dt)
    if not durs:
        durs = [1.0 / max(1e-9, speed)]  # single-frame edge case
    with list_path.open("w", encoding="utf-8") as f:
        for p, dur in zip(files[:-1], durs):
            f.write(f"file '{p.resolve().as_posix()}'\n")
            f.write(f"duration {dur:.9f}\n")
        last_abs = files[-1].resolve().as_posix()
        # concat demuxer quirk: repeat last line to hold duration
        f.write(f"file '{last_abs}'\n")
        f.write(f"file '{last_abs}'\n")

def side_by_side_vfr_from_folders(
    vision_frames_dir: Union[str, Path],
    sonar_frames_dir: Union[str, Path],
    out_mp4: Union[str, Path],
        *,
    out_height: int = 720,        # output height; width auto-kept for aspect
    crf: int = 18,
    preset: str = "slow",
    min_dur: float = 1/120,       # clamp tiny gaps
    max_dur: float = 1.0,         # clamp huge gaps
    speed: float = 1.0,           # 0.5 = slow-mo, 2.0 = fast
) -> Path:
    """
    Build a side-by-side MP4 aligned by real timestamps from two folders of frames
    named '<ns>.jpg'. Uses FFmpeg concat (VFR) + tpad for start alignment.
    """
    vision_frames_dir = Path(vision_frames_dir)
    sonar_frames_dir  = Path(sonar_frames_dir)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # 1) Collect files + timestamps
    v_files, v_ts = _collect_ts_sorted(vision_frames_dir)
    s_files, s_ts = _collect_ts_sorted(sonar_frames_dir)

    # 2) Build per-stream concat lists (VFR)
    v_list = out_mp4.with_suffix(".vision.list.txt")
    s_list = out_mp4.with_suffix(".sonar.list.txt")
    _write_concat_list(v_files, v_ts, v_list, min_dur=min_dur, max_dur=max_dur, speed=speed)
    _write_concat_list(s_files, s_ts, s_list, min_dur=min_dur, max_dur=max_dur, speed=speed)

    # 3) Compute start offset and pad the earlier stream
    v_start = v_ts[0] / 1e9
    s_start = s_ts[0] / 1e9
    pad_v = max(0.0, (max(v_start, s_start) - v_start) / max(1e-9, speed))
    pad_s = max(0.0, (max(v_start, s_start) - s_start) / max(1e-9, speed))

    # 4) FFmpeg command:
    #    - two concat inputs
    #    - pad early stream with tpad=start_duration=<pad>
    #    - scale both to the same height, preserve aspect (width = -2)
    #    - hstack side by side, stop at the shorter stream
    v_in = v_list.resolve().as_posix()
    s_in = s_list.resolve().as_posix()
    out = out_mp4.resolve().as_posix()

    pad_v_str = f"tpad=start_duration={pad_v}:start_mode=clone" if pad_v > 0 else "null"
    pad_s_str = f"tpad=start_duration={pad_s}:start_mode=clone" if pad_s > 0 else "null"

    filter_complex = (
        f"[0:v]{pad_v_str},scale=-2:{out_height}[v0];"
        f"[1:v]{pad_s_str},scale=-2:{out_height}[v1];"
        f"[v0][v1]hstack=inputs=2:shortest=1[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", v_in,
        "-f", "concat", "-safe", "0", "-i", s_in,
        "-fps_mode", "vfr",
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-movflags", "+faststart",
        out,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ Wrote side-by-side: {out_mp4}")

    # cleanup
    try:
        v_list.unlink()
        s_list.unlink()
        print(f"🧹 Deleted temporary files: {v_list.name}, {s_list.name}")
    except Exception as e:
        print(f"⚠️ Could not delete temporary files: {e}")

    return out_mp4
