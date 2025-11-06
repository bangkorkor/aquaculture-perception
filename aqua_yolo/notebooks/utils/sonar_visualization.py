# sonar_visualization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

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

# ========== custom enhancer to match public sonar dataset =================

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



# --- raw plot ----------------------------------------------------------------

def plot_raw_frame(M: np.ndarray, frame_index: int, cfg: dict):
    """
    Plot the raw sonar frame 
    """
    # --- orientation operations directly ---
    Z = M.copy()
    if cfg.get("transpose_M", False): # returns cfg[key] if it exists, false otherwise
        Z = Z.T     # we transpose the matrix (swap H and W)
    if cfg.get("flipX_M", False):
        Z = Z[::-1, :]  # we flip the beam angles 
    if cfg.get("flipY_M", False):
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
    if cfg.get("flipX_M", False):
        Z = Z[::-1, :]  # we flip the beam angles 
    if cfg.get("flipY_M", False):
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
    if cfg.get("flipY_M", False):
        Z = Z[::-1, :]
    if cfg.get("flipX_M", False):
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
