# SOLAQUA Sonar dataprocessing

## How is processing of sonar data done

**THIS IS NOT DONE! LOOK OVER THIS!**
Show the flow here.

## Table of Contents
1. [What we want to achieve](#1-what-we-want-to-achieve)
2. [Extracting Data](#1-extracting-data)
3. [Plotting Polar Data](#2-plotting-polar-data)
4. [Sonar Enhancer](#3-sonar-enhancer)
5. [Polar to Cartesian](#4-polar-to-cartesian)

## 1 What we want to achieve

Goal is to process sonar data to information we want to do something with. 

### 1.1 CFC style replication

I am first trying to process the data into .png images that look like the ones from the cfc dataset. This way I can test my cfc trained model and see if it can recognice any fish.

<div align="center">
<img src="https://github.com/user-attachments/assets/3934a250-ac09-4802-b28b-0e17beacb26f" width="400" />
<img src="https://github.com/user-attachments/assets/0897df09-db8b-4563-a21f-eefff00b2cf4" width="400" />
<p><em>Figure: My processed sonar frame (left) vs. CFC-style frame (right).</em></p>
</div>


### 1.2 Later scope

I porbably want to process the data into my own styled .pngs to make it simple to label the net. Then i can use the labeled net-data to train my own model and see if it is able to recognize any net. 

## 2 Extracting Data

### 2.1 Bag Files

**Explain the timestamp stuff and what is in the bags**

- `*_data.bag` → contains **sensor data** (ROS bag format).
- `*_video.bag` → contains **video images** (ROS bag format).

The dataset used is **SOLAQUA**, available from [SINTEF Open Data](https://data.sintef.no/feature/fe-a8f86232-5107-495e-a3dd-a86460eebef6).



### 2.2 Load the data

- First we set the `_video.bag` path as `VIDEO_BAG` and we select the frame we want to save as `VIDEO_FRAME`. These are feed to the `load_sonoptix_frame_from_bag` function.

```python
DATA_BAG = Path("...")
VIDEO_BAG = Path("...")
VIDEO_FRAME = 1000

M_raw, t_ns = load_sonoptix_frame_from_bag(VIDEO_BAG, VIDEO_FRAME) # returns M_raw and timestamp
```

  

### 2.3 Convert to raw data on right format

**inside load_sonoptix_frame_from_bag function:**

- AnyReader from rosbag opens the .bag and inspects all topic streams. It picks the first connection whose topic is "/sensor/sonoptix_echo/image" (the TOPIC variable) or whose message type is "sensors/msg/SonoptixECHO" (the MSGTYPE variable). Inside the bag there should only be one connection that matches this.

  
```python
with AnyReader([bag_path]) as r:
	conns = [c for c in r.connections if c.topic == TOPIC or c.msgtype == MSGTYPE]
```

  

- We then iterate messages (within this topic? idk) until we reach the requested index (the VIDEO_FRAME). _normalize_tuple(tup) makes it robust: it extraxcts the message timestamp: t_ns, raw: the raw serialized bytes, and conn: the connection object.

  
```python
for i, tup in enumerate(r.messages([conn])):
	if i == index:
		t_ns, raw, conn2 = _normalize_tuple(tup)

```

  

- Then we use AnyReader's function deserialize, to turn raw bytes into a typed message object of the right type. We can access the payload via msg.array_data. We then extract numerica data and cast it to float32 as numpy array.

  

```python
msg = r.deserialize(raw, conn2.msgtype)
data = np.asarray(msg.array_data.data, dtype=np.float32)
```

  

- Now we determine the shape and reshape it to 1024x256. Mark that the code here is not the best. But the size should always be 1024*256 i think, so it will correct shape.

  

```python

dims = getattr(msg.array_data.layout, "dim", [])
H = int(dims[0].size) if len(dims) > 0  else  None
W = int(dims[1].size) if len(dims) > 1  else  None
...
return data.reshape(1024, 256)

```

- We now have the shape (1024, 512) with raw recorded values. These values seem to already be discretized or scaled to the range 0-59 (?). We also return the timestamp of the frame.

**TODO:** we can also get the time in utc

## 3 Plotting Polar Data

  
### 3.1 Plotting the raw data
- plot_raw_frame and plot_enhanced_frame takes as input M_raw, VIDEO_FRAME, SONAR_IMAGES_CONFIG: config that we have defined in a code-block higher up. **TODO:** would also be nice to take in the timestamp for later, when I want to export .pngs.

  

inside the plot_raw_frame and plot_enhanced_frame:**this is not important and can be removed!!**

- In the config we have the option to do operations on the M_raw matrix. The operations are described in the comments.
```python
Z = M.copy()
if cfg.get("transpose_M", False): # returns cfg[key] if it exists, false otherwise
	Z = Z.T # we transpose the matrix (swap H and W)
if cfg.get("flipX_M", False):
	Z = Z[::-1, :] # we flip the beam angles
if cfg.get("flipY_M", False):
	Z = Z[:, ::-1] # we flip the range angles
```

  

- For enchanced version we apply, enhancements. See own section.
```python
# Default if none provided → enhance_intensity
if enhancer is  None:
	enhancer = enhance_intensity
# Apply enhancer
Z_enh = enhancer(Z, cfg)

```

- Axes mapping (for imshow extent). extent=(x_min, x_max, y_min, y_max). **Note: HOW do we now what is the right FOV and depth? We have seen this in the online documentation for the sonoptix echo. just search on goolge. **

```python
theta_min = -0.5 * float(cfg["fov_deg"])
theta_max = +0.5 * float(cfg["fov_deg"])
extent = (theta_min, theta_max, float(cfg["range_min_m"]), float(cfg["range_max_m"]))
```

- Make figure, then render:
```python
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

```
- Apply the desired cropping:

```python
# apply display crop
ax.set_ylim(cfg["range_min_m"], cfg["display_range_max_m"])
```
- We now see the images in polar form but stretched onto a rectangular grid. Each pixel in M is a point in polar space.

  
  

## 4 Sonar Enhancer:

- We have two enhancers. the default enhance_intensity, and the one designed to look cfc images: enhance_cfc_style
### 4.1 Enhance intensity
 
- enhance_intensity: uses TVG, scaling and percentile normalization.

  
### 4.1 Enhance CFC style 

This is to try to match the look of the cfc_dataset.
- enhance_cfc_style: uses: sanitize, TVG: geometric spreading + absorption, Per-range background flattening, Log/dB, contrast stretch to [0,1], Gamma: lift highlights a bit, noise, hard white cap.



## 5 Polar to Cartesian

### 5.1 Cone Plot

What i want: I want a cone view where:

- The data has a fov and the image should correctly display this. How does this work? Cartesian vs polar. We see the image in cartesian coordinaes, but then it has to be a cone view, to get the angels right.

  
  

**Cone display**:

- First we can agian transpose and enhance it if needed.

  

- Then we set the actual limits of the cone and then the actual specs we want. We feed this to the cone_rasterizer_display_cell function.

  

```python
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

```

- See section about rasterizer for the code there. This will return. a "D float array in XY meters. A tuple of the extent fo imshow. amid and ahalf, The center angle and the half-width of the user window.


- We now create the figure and cmap. set_bad means NaNs will render as black. the vmin and vmax are mostly for the colorbar.

- We create the image, set axis and add the colorbar:

 
```python
im = ax.imshow(
	cone,
	origin="lower",
	extent=(x_min, x_max, y_min, y_max),
	aspect="equal",
	cmap=cmap,
	vmin=vmin, vmax=vmax,
)
```

  
  
  
  
### 5.2 Rasterization
**cone_rasterizer_display_cell**:
**chat is used a lot here and i dont get this stuff fully yet**


- First we validate if the config-parameters for the angle and depth are valid.

 
- We get the mid angle. And also the half of the mid angle:

```python
amid = 0.5*(amin + amax) # center angle
ahalf = 0.5*(amax - amin) # half angular span ≥ 0
```


- Build a symmetric output grid in XY. We always cover the radius up to cv_rmax (r1). The inner edge (below cv_rmin) will be masked, not cropped, so it stays circular.

```python
y_min, y_max = 0.0, r1
x_span = y_max * sin(ahalf)
x_min, x_max = -x_span, +x_span

ys = linspace(y_min, y_max, img_h, endpoint=False)
xs = linspace(x_min, x_max, img_w, endpoint=False)
Xc, Yc = meshgrid(xs, ys)

```

- Get “relative” angles and ranges.

```python
theta_rel = deg(atan2(Xc, Yc)) # in the *centered view frame*
rng = hypot(Xc, Yc)
```
- ...


- Interpolating avoids blocky artifacts and respects sub-pixel geometry when mapping polar → Cartesian.

```python
r0i = floor(r_idx); r1i = clip(r0i+1, 0, H-1)
c0i = floor(beam_idx); c1i = clip(c0i+1, 0, W-1)

  
fr = r_idx - r0i # vertical fraction
fc = beam_idx - c0i # horizontal fraction

  
# Blend four neighbors
Z00 = Z[r0i, c0i]; Z01 = Z[r0i, c1i]
Z10 = Z[r1i, c0i]; Z11 = Z[r1i, c1i]
top = Z00*(1-fc) + Z01*fc
bot = Z10*(1-fc) + Z11*fc
Zi = top*(1-fr) + bot*fr
```

### 6 Extracting data

1 we make frames
2 we use the frame folder to make a video based on the timestamps (every frame has the ns timestamp as its name). We use Variable frame rate (VFR) from the ffmpeg library to time stuff. 


