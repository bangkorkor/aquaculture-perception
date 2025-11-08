net_sonar_labelingview_config = {
    # ===== data mapping. From matrix to real-values
    "fov_deg": 120.0,     # field of view. in degrees. 0 in middle (+-60)
    "range_min_m": 0.0, 
    "range_max_m": 20.0,    # depth of data

    "transpose_M": False,   # we transpose the matrix (swap H and W)
    "flipX_m": True,    # we flip the beam angle, True here is correct, dont know why
    "flipY_m": False,    # we flip the range angles,

    # ===== default enhancement - 
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
    # base
    "denoise_kernel": 3,
    "cluster_kernel": 5,
    "cluster_blend": 0.4,

    "p_low": 5.0,
    "p_high": 99.5,
    "gamma": 0.7,
    "edgy_boost": 0.25,

    # Sharpen
    "sharpen_enabled": True,
    "sharpen_amount": 1.2,

    # Local contrast
    "clahe_enabled": False,     # turn on if we want strong local contrast
    "clahe_clip": 3.0,

    # Cluster-only boost
    "cluster_mask_enabled": True,
    "cluster_mask_thr": 0.25,
    "cluster_mask_strength": 0.4,

    # Rim effect
    "rim_enabled": True,
    "rim_strength": 0.2,
    


    # ===== visualization polar =============
    "cmap_raw": "viridis",
    "cmap_enh": "gray",
    "figsize": (6, 5.6),
    "display_range_min_m": 0.2, # how deep we show 
    "display_range_max_m": 5.0, # how deep we show

    # ===== cone view ===========
    "img_w": 1200,
    "img_h": 700,
    "bg_color": "green",
    "rotate_deg": 0.0,  
    "coneview_range_min_m": 0.2,
    "coneview_range_max_m": 8.0,
    "coneview_angle_min_deg": -45.0,
    "coneview_angle_max_deg": 45.0,
    }
