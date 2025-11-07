cfc_gray_config = {
    # ===== data mapping. From matrix to real-values
    "fov_deg": 120.0,     # field of view. in degrees. 0 in middle (+-60)
    "range_min_m": 0.0, 
    "range_max_m": 20.0,    # depth of data

    "transpose_M": False,   # we transpose the matrix (swap H and W)
    "flipX_m": True,    # we flip the beam angle, True here is correct, dont know why
    "flipY_m": False,    # we flip the range angles,

    "enhancer_type":"cfc_style", # "default" is the enhance_intensity(), and custom is the cfc_style is the enhance_sonar_dataset_style()

    # ===== default enhancement
    "enh_scale": "db",            # db | log | sqrt | lin
    "enh_tvg": "amplitude",       # amplitude | none
    "enh_alpha_db_per_m": 0.0,
    "enh_eps_log": 1e-5,
    "enh_r0": 1e-6,
    "enh_p_low": 1.0,
    "enh_p_high": 99.5,
    "enh_gamma": 0.9,
    "enh_zero_aware": True,

    # ===== custom enhancement
    "ds_geo_exponent": 1.0,      # geometric spreading comp (× r^g)
    "ds_alpha_db_per_m": 0.0,    # absorption gain [dB/m]; try 0.02..0.06 if needed
    "ds_alpha_r0": 0.0,          # start distance for absorption comp

    "ds_bg_percentile": 60.0,    # per-row baseline removal
    "ds_bg_scale": 1.0,

    "ds_eps_log": 1e-5,          # log epsilon
    "ds_p_low": 2.0,             # robust stretch
    "ds_p_high": 99.7,           # reduces extreme stretching, less whitness
    "ds_gamma": 0.9,             # tone curve (0.85–1.0 typical)
    "ds_max_white_cap": 0.85,       # max cap for whiteness. 1 keeps completely white

    "ds_soft_floor": 0.25,       # lifts background toward mid-gray (0 disables)

    "ds_noise_enabled": True,
    "ds_noise_std": 0.10,       # try 0.01–0.10
    "ds_noise_seed": None,      # or an int for reproducibility


    # ===== visualization polar
    "cmap_raw": "viridis",
    "cmap_enh": "gray",
    "figsize": (6, 5.6),
    "display_range_min_m": 0.2, # how deep we show 
    "display_range_max_m": 5.0, # how deep we show

    # ===== cone view 
    "img_w": 1200,
    "img_h": 700,
    "bg_color": "#4b4b4b",
    "rotate_deg": 0.0,  
    "coneview_range_min_m": 0.2,
    "coneview_range_max_m": 8.0,
    "coneview_angle_min_deg": -45.0,
    "coneview_angle_max_deg": 45.0,
}
