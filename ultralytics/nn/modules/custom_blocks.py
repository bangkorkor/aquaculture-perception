from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock


class UWYOLO_ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act='SiLU', bias=False):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        if act is None:
            self.act = nn.Identity()
        elif isinstance(act, str) and act.lower() == 'gelu':
            self.act = nn.GELU()
        elif isinstance(act, str) and act.lower() == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:  # default SiLU
            self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PConv(nn.Module):
    """
    Partial Convolution (PConv): apply spatial conv on a fraction r of channels,
    pass the remaining (1-r) channels unchanged, then concatenate.

    Args:
        c (int): in/out channels
        k (int): kernel size (paper uses 3)
        s (int): stride (paper uses 1 inside Faster block)
        r (float): partial ratio (paper uses 1/4)
    """
    def __init__(self, c: int, k: int = 3, s: int = 1, r: float = 0.25):
        super().__init__()
        assert 0.0 < r <= 1.0
        self.c = c
        self.cp = max(1, int(round(c * r)))  # number of channels to convolve
        self.uc = c - self.cp                 # unchanged channels
        self.s = s
        self.k = k
        # Only convolve the first cp channels
        self.conv = UWYOLO_ConvBNAct(self.cp, self.cp, k=k, s=s, act='SiLU')

    def forward(self, x):
        x1, x2 = torch.split(x, [self.cp, self.uc], dim=1)
        y1 = self.conv(x1)
        # stride > 1 is not used in the paper's block, but handle gracefully
        if self.s > 1 and y1.shape[-2:] != x2.shape[-2:]:
            x2 = F.avg_pool2d(x2, kernel_size=self.s, stride=self.s)
        return torch.cat([y1, x2], dim=1)


class FasterBlock(nn.Module):
    """
    FasterNet block with same in/out channels.
    Accepts (c1, c2=None, r=0.25) to be YAML/Ultralytics-friendly.
    """
    def __init__(self, c1: int, c2: int = None, r: float = 0.25):
        super().__init__()
        c = c1 if c2 is None else c2
        assert c == c1, "FasterBlock expects c2==c1 (same in/out channels)."
        self.pconv = PConv(c, k=3, s=1, r=r)
        self.expand = UWYOLO_ConvBNAct(c, 2 * c, k=1, s=1, act='RELU')
        self.project = UWYOLO_ConvBNAct(2 * c, c, k=1, s=1, act=None)

    def forward(self, x):
        y = self.project(self.expand(self.pconv(x)))
        return x + y



class ChannelShuffle(nn.Module):
    """Channel shuffle utility used by GSConv."""
    def __init__(self, groups: int = 2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.size()
        g = self.groups
        assert c % g == 0
        x = x.view(n, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(n, c, h, w)


class GSConv(nn.Module):
    """
    GSConv: parallel SC and DSC branches on the same input, then concat + channel shuffle.
    Matches the Slim-neck by GSConv description used by the paper’s LC2f.  (Fig. 5)
    """
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        # ensure even out channels for clean 2-way shuffle
        self.c2 = c2 if c2 % 2 == 0 else c2 + 1
        mid = self.c2 // 2

        # SC branch (standard 1x1 conv)
        self.sc = UWYOLO_ConvBNAct(c1, mid, k=1, s=1, act='SiLU')

        # DSC branch (depthwise kxk on x, then 1x1)
        self.dw = UWYOLO_ConvBNAct(c1, c1, k=k, s=s, g=c1, act='SiLU')   # depthwise on input x
        self.pw = UWYOLO_ConvBNAct(c1, mid, k=1, s=1, act='SiLU')        # pointwise

        self.shuffle = ChannelShuffle(groups=2)

        # If caller asked for odd c2, trim back after shuffle
        self.trim = (self.c2 != c2)

    def forward(self, x):
        a = self.sc(x)          # SC(x)
        b = self.pw(self.dw(x)) # DSC(x)
        y = torch.cat([a, b], dim=1)  # (B, c2_even, H, W)
        y = self.shuffle(y)
        if self.trim:
            y = y[:, : self.c2 - 1, ...]  # drop one channel to match requested c2
        return y



class LC2f(nn.Module):
    """
    Lightweight-C2f: C2f-style split-and-merge where:
      - the inner bottleneck is replaced by FasterBlock
      - the final conv is replaced by GSConv
    Args:
        c1 (int): input channels
        c2 (int): output channels
        n (int): number of inner blocks
        shortcut (bool): residual inside FasterBlock is internal; outer shortcut not required
        r (float): PConv ratio for FasterBlock
    """
    def __init__(self, c1: int, c2: int, n: int = 2, shortcut: bool = True, r: float = 0.25):
        super().__init__()
        c_hidden = c2 // 2
        # stem expand to two branches like C2f
        self.cv1 = UWYOLO_ConvBNAct(c1, c_hidden, k=1, s=1)
        self.cv2 = UWYOLO_ConvBNAct(c1, c_hidden, k=1, s=1)
        # stack FasterBlocks on the second branch
        self.blocks = nn.Sequential(*[FasterBlock(c_hidden, r=r) for _ in range(n)])
        # concatenate [branch1, branch2] -> GSConv to c2
        self.fuse = GSConv(2 * c_hidden, c2, k=3, s=1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y2 = self.blocks(y2)
        y = torch.cat([y1, y2], dim=1)
        return self.fuse(y)
    

# ---------------------------------------------------------------------------
# AquaYOLO custom blocks 
# ---------------------------------------------------------------------------
# Basic Conv + BN + Activation (used everywhere)
class AQUAYOLO_ConvBNAct(nn.Module):
    """
    Conv2d → BatchNorm2d → Activation (default: ReLU)
    Works like Ultralytics Conv block but lighter for custom modules.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):    # We try kernel size 1x1 to limit parameters, paper does not specify
        super().__init__()
        if p is None:
            p = k // 2 # This will give padding = 1
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False) # Bias is off because of the following BN
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Residual Block (AquaYOLO backbone)
# class AquaResidualBlock(nn.Module):
#     """
#     Two AQUAYOLO_ConvBNAct blocks + skip connection.
#     Paper: AquaYOLO residual backbone (Fig. 1)  The paper does not explicitly mention BN but we use it. 
#     """
#     def __init__(self, c1, c2, stride=1):
#         super().__init__()
#         self.conv1 = AQUAYOLO_ConvBNAct(c1, c2, k=3, s=stride)
#         self.conv2 = AQUAYOLO_ConvBNAct(c2, c2, k=3, s=1, act=False)
#         self.proj = None
#         if stride != 1 or c1 != c2:
#             self.proj = AQUAYOLO_ConvBNAct(c1, c2, k=1, s=stride, act=False)
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, x):
#         identity = x if self.proj is None else self.proj(x)
#         out = self.conv2(self.conv1(x))
#         return self.act(out + identity)

class AquaResidualBlock(nn.Module):
    """
    Residual block as in the paper: [37] in paper goes in detail.
      - Two 3×3 Conv2d layers (stride=1, padding=1), no BN inside the block
      - Add skip (identity) to the stacked conv output
      - Apply ReLU AFTER the addition

    Notes from the paper:
      - The residual block contains two convolutional layers.
      - Each conv layer uses 3×3 kernels; ReLU is applied after the skip-add.
      - BatchNorm is used in the *separate conv layer that follows the block*,
        not inside the block itself. :contentReference[oaicite:0]{index=0}
    """
    def __init__(self, c1: int, c2: int, s: int = 1, k: int = 3, p: int = None, bias: bool = True): # Paper does not specify bias, but because it is not followed directly by BN we use it
        super().__init__()
        if p is None:
            # 'same' padding for odd k (k=3 → p=1)
            p = k // 2

        # main path
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, bias=bias) # only here we set the stride ok
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=k, stride=1, padding=p, bias=bias)       

        # projection for shape change, this makes it so that the block gets stride=s
        self.proj = None
        if s != 1 or c1 != c2:
            self.proj = nn.Conv2d(c1, c2, kernel_size=1, stride=s, padding=0, bias=bias)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = self.conv2(self.conv1(x))
        out = out + identity
        return self.act(out)



class CAFS(nn.Module):
    def __init__(self, c, hidden=None):
        super().__init__()
        h = hidden or c

        # pre-convs on Fa and Fb (as you already have)
        self.pre_a = AQUAYOLO_ConvBNAct(c, c, k=1, s=1)
        self.pre_b = AQUAYOLO_ConvBNAct(c, c, k=1, s=1)

        # central mix from concat -> h (C channels)
        self.mix = nn.Sequential(
            AQUAYOLO_ConvBNAct(2 * c, h, k=1, s=1),     # 2*c because of concat
            AQUAYOLO_ConvBNAct(h,     c, k=1, s=1),
        )

        # RIGHT path: three plain convs fed from h (C -> C), then CBR to 2ch, softmax
        self.right_plain = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, bias=True),        
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True),
        )
        self.right_cbr = AQUAYOLO_ConvBNAct(c, 2, k=1, s=1)    # produces 2-channel logits

        # LEFT gate: from h
        self.left = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, bias=True),
            AQUAYOLO_ConvBNAct(c, c, k=1, s=1),
        )

    def forward(self, Fa, Fb):
        Fa_p = self.pre_a(Fa)
        Fb_p = self.pre_b(Fb)
        cat  = torch.cat([Fa_p, Fb_p], dim=1)

        h = self.mix(cat)                                     # [B, C, H, W]

        ab_feat = self.right_plain(h)                         # now expects C, gets C
        logits  = self.right_cbr(ab_feat)                     # [B, 2, H, W]
        Wa, Wb  = torch.softmax(logits, dim=1).chunk(2, dim=1)

        Wf = torch.sigmoid(self.left(h))                      # [B, C, H, W]

        fused = Wa * Fa + Wb * Fb
        out   = fused + (Wf * h)
        return out



class FAU(nn.Module):
    """
    FAU (Feature Alignment Unit):
      - Resize Fb to Fa's H×W using bilinear
      - 3×3 Conv + BN + ReLU
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = AQUAYOLO_ConvBNAct(c_in, c_out, k=1, s=1)   # does not specify kernel size, we try k=1

    def forward(self, x, target_hw):
        x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return self.conv(x)

    
class DSAM(nn.Module):
    """
    DSAM (Fig. 2-style) with clear naming.

    Inputs
      Fa : [B, C_a, H, W]  (target scale)
      Fb : [B, C_b, h, w]  (adjacent scale)

    Wiring
      Left:
        Fa --FAU--> L_a
        Fb --FAU--> L_b (the output must match C_a dimentions so that we can do ultiplication)
        left_mul = L_a ⊗ L_b                       # element-wise multiply

      Right CAFS branch:
        Fa: CBR -> FAU(align/refresh) -> CBR       # stays at (H,W), ends at C_a
        Fb: CBR -> FAU(align to Fa) -> CBR         # becomes (H,W), C_a
        cafs_out = CAFS(Fa_path, Fb_path) -> CBR   # post CBR per fig

      Merge & final:
        added = left_mul ⊕ cafs_out                # element-wise add
        out   = 1×1 Conv (+BN+ReLU)                # your requested final layer
    """
    def __init__(self, ch_in, ch_b=None):
        super().__init__()
        # Accept C_a alone or (C_a, C_b)
        if isinstance(ch_in, (list, tuple)):
            C_a, C_b = ch_in
        else:
            C_a, C_b = ch_in, (ch_b if ch_b is not None else ch_in)

        # ---------- Left path ( FAU on Fa and Fb, then multiply them) ----------
        self.left_fau_A = FAU(c_in=C_a, c_out=C_a)
        self.left_fau_B = FAU(c_in=C_b, c_out=C_a)

        # ---------- Right path → CAFS ----------
        # Fa sub-path into CAFS: CBR -> FAU -> CBR (kept even if size already matches, to mirror fig)
        self.fa_pre_cbr   = AQUAYOLO_ConvBNAct(C_a, C_a, k=1, s=1)
        self.fa_align_fau = FAU(c_in=C_a, c_out=C_a)
        self.fa_post_cbr  = AQUAYOLO_ConvBNAct(C_a, C_a, k=1, s=1)

        # Fb sub-path into CAFS: CBR -> FAU(align to Fa) -> CBR
        self.fb_pre_cbr   = AQUAYOLO_ConvBNAct(C_b, C_b, k=1, s=1)
        self.fb_align_fau = FAU(c_in=C_b, c_out=C_a)          # also changes channels to C_a
        self.fb_post_cbr  = AQUAYOLO_ConvBNAct(C_a, C_a, k=1, s=1)

        # CAFS core (your fixed, figure-faithful version)
        self.cafs = CAFS(c=C_a)

        # post-CAFS CBR (yellow box under CAFS)
        self.cafs_out_cbr = AQUAYOLO_ConvBNAct(C_a, C_a, k=3, s=1)

        # final 1×1 conv
        self.final_conv = nn.Conv2d(C_a, C_a, kernel_size=1, bias=True)     # No Bn after so we use bias

    def forward(self, inputs):
        Fa, Fb = inputs
        H, W = Fa.shape[-2:]

        # ----- Left path -----
        L_a = self.left_fau_A(Fa, target_hw=(H, W))      # [B, C_a, H, W]
        L_b = self.left_fau_B(Fb, target_hw=(H, W))      # [B, C_a, H, W]
        left_mul = L_a * L_b                             # element-wise multiply

        # ----- Right path (Fa branch) -----
        fa1 = self.fa_pre_cbr(Fa)                         # [B, C_a, H, W]
        fa2 = self.fa_align_fau(fa1, target_hw=(H, W))    # [B, C_a, H, W]
        fa3 = self.fa_post_cbr(fa2)                       # [B, C_a, H, W]

        # ----- Right path (Fb branch) -----
        fb1 = self.fb_pre_cbr(Fb)                         # [B, C_b, h, w]
        fb2 = self.fb_align_fau(fb1, target_hw=(H, W))    # [B, C_a, H, W]
        fb3 = self.fb_post_cbr(fb2)                       # [B, C_a, H, W]

        # ----- CAFS + post CBR -----
        cafs_out = self.cafs(fa3, fb3)                    # [B, C_a, H, W]
        cafs_cbr_out = self.cafs_out_cbr(cafs_out)        # [B, C_a, H, W]

        # ----- Add left & right, then final 1×1 -----
        added = left_mul + cafs_cbr_out                   # element-wise add
        out = self.final_conv(added)                      # 1×1 conv only
        return out



# -----------------------------------------
# --------------- AGW-YOLOv8 --------------
# -----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F



# Simple Conv-BN-Act helper 

class AGW_ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act='SiLU', bias=False):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(c2)

        if act is None:
            self.act = nn.Identity()
        elif isinstance(act, str) and act.lower() == 'gelu':
            self.act = nn.GELU()
        elif isinstance(act, str) and act.lower() == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------------
# CBAM (Backbone attention)
# -------------------------
class AGW_ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP implemented with 1x1 convs
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.mlp(self.avg_pool(x))
        m = self.mlp(self.max_pool(x))
        return self.sigmoid(a + m)


class AGW_SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # channel-wise avg and max -> 2ch map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(y))


class AGW_CBAM(nn.Module):
    """
    YAML-friendly: CBAM(c1, c2=None, reduction=16, spatial_kernel=7)
    Expects c2==c1 (no channel change).
    """
    def __init__(self, c1: int, c2: int = None, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        c2 = c1 if c2 is None else c2
        assert c1 == c2, "CBAM expects same in/out channels (c2==c1)."
        self.c2 = c1
        self.ca = AGW_ChannelAttention(c1, reduction=reduction)
        self.sa = AGW_SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# -------------------------
# AGW_GSConv (Neck lightweight conv)
# -------------------------


class AGW_ChannelShuffle(nn.Module):
    def __init__(self, groups: int = 2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.size()
        g = self.groups
        assert c % g == 0, f"channels ({c}) must be divisible by groups ({g})"
        x = x.view(n, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(n, c, h, w)


class AGW_GSConv(nn.Module):
    """
    Paper-style GSConv:
      y1 = fconv(Xin)                      # SC branch (1x1 conv, stride s)
      y2 = fdsc(y1) = PW(DW(y1))           # DSC branch applied to y1 (stride 1)
      y  = shuffle(cat(y1, y2))

    YAML-friendly: GSConv(c1, c2, k=3, s=1)
    """
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act: str = "SiLU"):
        super().__init__()

        # ensure even concat width so shuffle(2) works
        c2_even = c2 if (c2 % 2 == 0) else (c2 + 1)
        mid = c2_even // 2

        # SC: 1x1 conv produces mid channels, handles stride s (downsampling happens here)
        self.sc = AGW_ConvBNAct(c1, mid, k=1, s=s, act=act)

        # DSC: depthwise kxk then pointwise 1x1, BOTH operate on y1, stride=1
        self.dw = AGW_ConvBNAct(mid, mid, k=k, s=1, g=mid, act=act)
        self.pw = AGW_ConvBNAct(mid, mid, k=1, s=1, act=act)

        self.shuffle = AGW_ChannelShuffle(groups=2)

        self.trim = (c2_even != c2)
        self._c2_req = c2
        self.c2 = c2  # for Ultralytics bookkeeping

    def forward(self, x):
        y1 = self.sc(x)
        y2 = self.pw(self.dw(y1))
        y = torch.cat([y1, y2], dim=1)
        y = self.shuffle(y)
        if self.trim:
            y = y[:, :self._c2_req, ...]
        return y


# -------------------------
# SE + SE-C2f (Neck fusion blocks)
# -------------------------
class SEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = self.gate(self.fc2(w))
        return x * w



class BottleneckSE(nn.Module):
    """
    YOLOv8-style bottleneck: 3x3 -> 3x3, with optional shortcut,
    plus SE after the second conv.
    Mirrors Ultralytics v8 Bottleneck pattern (often 3x3 + 3x3).
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        se_reduction: int = 16,
        act: str = "SiLU",
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = AGW_ConvBNAct(c1, c_, k=3, s=1, g=1, act=act)      # 3x3
        self.cv2 = AGW_ConvBNAct(c_, c2, k=3, s=1, g=g, act=act)      # 3x3 
        self.se = SEAttention(c2, reduction=se_reduction)
        self.add = shortcut and (c1 == c2)

    def forward(self, x):
        y = self.se(self.cv2(self.cv1(x)))
        return x + y if self.add else y


class SEC2f(nn.Module):
    """
    SE-C2f: C2f wrapper, but internal blocks are YOLOv8-style BottleneckSE.

    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 2,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        se_reduction: int = 16,
        act: str = "SiLU",
    ):
        super().__init__()
        self.c2 = c2
        c_ = int(c2 * e)  # hidden channels like C2f

        # same as Ultralytics C2f: expand to 2*c_, split, then n blocks, then fuse
        self.cv1 = AGW_ConvBNAct(c1, 2 * c_, k=1, s=1, act=act)

        # IMPORTANT: in Ultralytics C2f, the inner Bottleneck often uses e=1.0
        # so it keeps channel width c_ through the bottleneck (no further squeeze).
        self.m = nn.ModuleList(
            BottleneckSE(c_, c_, shortcut=shortcut, g=g, e=1.0, se_reduction=se_reduction, act=act)
            for _ in range(n)
        )

        self.cv2 = AGW_ConvBNAct((2 + n) * c_, c2, k=1, s=1, act=act)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for block in self.m:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, 1))



#------------
# MAS-yolov11
#------------

import math
from typing import Sequence, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2PSA
from ultralytics.nn.modules.head import Detect


class MSDA(nn.Module):
    """
    Multi-Scale Dilated Attention (MSDA) via sliding-window attention with different dilation rates.

    Uses per-head local attention computed with unfold (3x3) at dilation r.
    Effective receptive fields become 3,5,7,9 for r=1,2,3,4. :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        c: int,
        heads: int = 4,
        dilation_rates: Sequence[int] = (1, 2, 3, 4),
    ):
        super().__init__()
        assert c % heads == 0, f"MSDA: channels {c} must be divisible by heads {heads}"
        self.c = c
        self.heads = heads
        self.d = c // heads

        # if user passes fewer/more rates than heads, cycle deterministically
        rates = list(dilation_rates)
        if len(rates) != heads:
            rates = [rates[i % len(rates)] for i in range(heads)]
        self.rates = rates

        self.qkv = nn.Conv2d(c, 3 * c, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # [B, heads, d, H, W]
        q = q.view(b, self.heads, self.d, h, w)
        k = k.view(b, self.heads, self.d, h, w)
        v = v.view(b, self.heads, self.d, h, w)

        outs: List[torch.Tensor] = []
        hw = h * w

        for hi, rate in enumerate(self.rates):
            qh = q[:, hi].reshape(b, self.d, hw)            # [B, d, HW]
            kh = k[:, hi]                                   # [B, d, H, W]
            vh = v[:, hi]                                   # [B, d, H, W]

            # Unfold local neighborhoods (3x3) with dilation=rate, padding=rate → HW locations
            k_unf = F.unfold(kh, kernel_size=3, dilation=rate, padding=rate, stride=1)  # [B, d*9, HW]
            v_unf = F.unfold(vh, kernel_size=3, dilation=rate, padding=rate, stride=1)  # [B, d*9, HW]

            k_unf = k_unf.view(b, self.d, 9, hw)  # [B, d, 9, HW]
            v_unf = v_unf.view(b, self.d, 9, hw)

            # attention scores: dot(q, k_patch) over channel dim d → [B, 9, HW]
            scores = (qh.unsqueeze(2) * k_unf).sum(dim=1)  # [B, 9, HW]
            attn = scores.softmax(dim=1)

            # weighted sum of v patches → [B, d, HW]
            out = (v_unf * attn.unsqueeze(1)).sum(dim=2)   # [B, d, HW]
            outs.append(out.view(b, self.d, h, w))

        y = torch.cat(outs, dim=1)          # [B, C, H, W]
        y = self.proj(y)
        y = self.act(self.bn(y))
        return x + y


class C2PSA_MSDA(nn.Module):
    """
    C2PSA_MSDA = YOLOv11's C2PSA + MSDA refinement.
    The paper’s intent is embedding MSDA into C2PSA’s attention path; this wrapper is a clean
    drop-in that preserves Ultralytics wiring while adding MSDA. 
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        heads: int = 4,
        dilation_rates: Sequence[int] = (1, 2, 3, 4),
    ):
        super().__init__()
        self.base = C2PSA(c1, c2)
        self.msda = MSDA(c2, heads=heads, dilation_rates=dilation_rates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.msda(self.base(x))


class ASFF(nn.Module):
    """
    Adaptive Spatial Feature Fusion for one output level l.
    F_l(i,j) = a*x0->l + b*x1->l + g*x2->l with softmax weights from 1x1 convs. 
    """

    def __init__(self, ch: Sequence[int], level: int):
        super().__init__()
        assert len(ch) == 3, "ASFF expects 3 feature levels (P3,P4,P5)"
        assert level in (0, 1, 2)
        self.level = level
        self.inter = ch[level]

        # Build rescale paths into 'inter' channels + correct spatial size for this level
        self.p = nn.ModuleList([self._make_path(i, ch[i]) for i in range(3)])

        # Control parameters λ via 1x1 conv → softmax over {0,1,2} 
        self.w = nn.ModuleList([nn.Conv2d(self.inter, 1, 1) for _ in range(3)])

    def _make_path(self, src_level: int, in_ch: int) -> nn.Module:
        # target spatial scale by level:
        # level 0 ~ P3 (largest), level 1 ~ P4, level 2 ~ P5 (smallest)
        if self.level == 0:
            if src_level == 0:   # P3 -> P3
                return Conv(in_ch, self.inter, 1, 1)
            if src_level == 1:   # P4 -> up2
                return nn.Sequential(Conv(in_ch, self.inter, 1, 1), nn.Upsample(scale_factor=2, mode="nearest"))
            # P5 -> up4
            return nn.Sequential(Conv(in_ch, self.inter, 1, 1), nn.Upsample(scale_factor=4, mode="nearest"))

        if self.level == 1:
            if src_level == 0:   # P3 -> down2
                return Conv(in_ch, self.inter, 3, 2)
            if src_level == 1:   # P4 -> P4
                return Conv(in_ch, self.inter, 1, 1)
            # P5 -> up2
            return nn.Sequential(Conv(in_ch, self.inter, 1, 1), nn.Upsample(scale_factor=2, mode="nearest"))

        # self.level == 2
        if src_level == 2:       # P5 -> P5
            return Conv(in_ch, self.inter, 1, 1)
        if src_level == 1:       # P4 -> down2
            return Conv(in_ch, self.inter, 3, 2)
        # P3 -> down4 (two downsamples)
        return nn.Sequential(
            Conv(in_ch, self.inter, 3, 2),
            Conv(self.inter, self.inter, 3, 2),
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x0 = self.p[0](x[0])
        x1 = self.p[1](x[1])
        x2 = self.p[2](x[2])

        w0 = self.w[0](x0)
        w1 = self.w[1](x1)
        w2 = self.w[2](x2)
        ws = torch.softmax(torch.cat([w0, w1, w2], dim=1), dim=1)  # [B,3,H,W]

        return ws[:, 0:1] * x0 + ws[:, 1:2] * x1 + ws[:, 2:3] * x2


class ASFFHead(Detect):
    """
    Must match parse_model() call:
      ASFFHead(nc, reg_max, end2end, ch_list)
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)  # IMPORTANT: positional to match your Detect

        assert len(ch) == 3, f"ASFFHead expects 3 feature maps (P3,P4,P5). Got ch={ch}"
        ch = list(ch)

        self.asff0 = ASFF(ch, level=0)
        self.asff1 = ASFF(ch, level=1)
        self.asff2 = ASFF(ch, level=2)

    def forward(self, x):
        # x is [p3, p4, p5]
        x = [self.asff0(x), self.asff1(x), self.asff2(x)]
        return super().forward(x)










