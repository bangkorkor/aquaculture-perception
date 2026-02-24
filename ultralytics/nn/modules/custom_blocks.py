from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
from .block import Bottleneck, C3, C3k2

from typing import Optional
from einops import rearrange




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
    
class UWConvBN(nn.Module):
    """Conv + BN (no activation). Ultralytics-YAML friendly."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, bias=False):
        super().__init__()
        self.m = UWYOLO_ConvBNAct(c1, c2, k=k, s=s, p=p, g=g, act=None, bias=bias)

    def forward(self, x):
        return self.m(x)
    
class UWConv(nn.Module):
    """Conv only (no BN, no activation)."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, bias=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=bias)

    def forward(self, x):
        return self.conv(x)
    


class PConv(nn.Module):
    # partial convolutional block (PConv) per UW-YOLOv8. Uses UWConv (conv only, no BN, no act)

    def __init__(self, c: int, k: int = 3, s: int = 1, r: float = 0.25):
        super().__init__()
        assert s == 1, "FasterNet PConv should use stride=1"
        self.cp = max(1, int(round(c * r)))
        # self.cp = c // 4 # original UW-YOLOv8 uses fixed 1/4, is this better?
        self.uc = c - self.cp
        self.conv = UWConv(self.cp, self.cp, k=k, s=1, bias=True)  # conv only

    def forward(self, x):
        x1, x2 = torch.split(x, [self.cp, self.uc], dim=1)
        y1 = self.conv(x1)
        return torch.cat([y1, x2], dim=1)
    



class FasterBlock(nn.Module):
    def __init__(self, c1: int, c2: int = None, r: float = 0.25):
        super().__init__()
        c = c1 if c2 is None else c2
        assert c == c1
        self.pconv = PConv(c, k=3, s=1, r=r)
        self.expand = UWYOLO_ConvBNAct(c, 2 * c, k=1, s=1, act='GELU')  # Conv+BN+GELU. FIG 2 uses Relu but text says GELU.
        self.project = UWConv(2 * c, c, k=1, s=1, bias=True)            # Conv only

    def forward(self, x):
        return x + self.project(self.expand(self.pconv(x)))




class ChannelShuffle(nn.Module):
    """ShuffleNet-style channel shuffle (transpose-based, no FLOPs)."""
    def __init__(self, groups: int = 2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.groups
        assert c % g == 0, f"channels ({c}) must be divisible by groups ({g})"
        x = x.view(b, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)


class GSConv(nn.Module):
    """
    GSConv: SC branch + DSC branch -> concat -> shuffle
    Compatible with calls like GSConv(c1,c2,k=5,s=1)
    """
    def __init__(self, c1: int, c2: int, k: int = 5, s: int = 1, act: str = "SiLU", k_sc: int = 3):
        super().__init__()
        assert c2 % 2 == 0, "GSConv expects even c2."
        mid = c2 // 2

        # SC branch (standard conv)
        self.sc = UWYOLO_ConvBNAct(c1, mid, k=k_sc, s=s, act=act)

        # DSC branch (depthwise kxk then pointwise 1x1)
        self.dw = UWYOLO_ConvBNAct(c1, c1, k=k, s=s, g=c1, act=act)   # depthwise
        self.pw = UWYOLO_ConvBNAct(c1, mid, k=1, s=1, act=act)        # pointwise

        self.shuffle = ChannelShuffle(groups=2)

    def forward(self, x):
        a = self.sc(x)
        b = self.pw(self.dw(x))
        return self.shuffle(torch.cat([a, b], dim=1))





class LC2f(nn.Module):
    """
    LC2f: drop-in C2f variant
      - same layout as Ultralytics C2f:
          cv1 -> split -> (block)x n -> concat -> cv2
      - replaces Bottleneck with FasterBlock
      - replaces final Conv with GSConv
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,   # kept for signature compatibility; not used
        g: int = 1,               # kept for signature compatibility; not used
        e: float = 0.5,
        r: float = 0.25,          # FasterBlock partial ratio
        gs_k: int = 3,            # GSConv depthwise kernel (paper might use 3/5)
    ):
        super().__init__()
        self.c = int(c2 * e)  # EXACTLY like C2f

        # stem conv then split into 2 chunks of size self.c
        self.cv1 = Conv(c1, 2 * self.c, k=1, s=1)

        # n FasterBlocks operating sequentially on the "second" branch output
        self.m = nn.ModuleList(FasterBlock(self.c, r=r) for _ in range(n))

        # concat of (2 + n) chunks -> GSConv to c2 (replaces C2f.cv2 Conv)
        self.cv2 = GSConv((2 + n) * self.c, c2, k=gs_k, s=1)

        # Keep these just so the constructor matches typical C2f YAML patterns
        self.n = n
        self.shortcut = shortcut
        self.g = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))     # [y0, y1], each (B, self.c, H, W)
        for m in self.m:
            y.append(m(y[-1]))                # append each intermediate output (C2f style)
        return self.cv2(torch.cat(y, 1))



# ---------------------------------------------------------------------------
# UODN custom blocks 
# ---------------------------------------------------------------------------

class CSMBBackBone1(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        assert c2 % 2 == 0
        c_ = c2 // 2

        # common 1x1 (keep spatial size)
        self.common = Conv(c1, c2, k=1, s=1, p=0)

        # branches: c2 -> c_
        self.l = Conv(c2, c_, k=3, s=1, p=1)
        self.r = Conv(c2, c_, k=3, s=1, p=1)
        self.m = Conv(c2, c_, k=3, s=1, p=1)

        # DB on mid: c_ -> c_ -> c_ with residual add
        self.db1 = Conv(c_, c_, k=3, s=1, p=1)
        self.db2 = Conv(c_, c_, k=3, s=1, p=1)

        # concat 3 maps => 3*c_
        self.out = Conv(3 * c_, c2, k=1, s=1, p=0)

    def forward(self, x):
        x = self.common(x)
        l_out = self.l(x)
        r_out = self.r(x)
        m_out = self.m(x)

        db_out = m_out + self.db2(self.db1(m_out))

        return self.out(torch.cat([l_out, db_out, r_out], dim=1))

class CSMBBackBone2(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        assert c2 % 2 == 0, "CSMB uses hidden width c2//2; requires even c2."
        c_ = c2 // 2

        self.common = Conv(c1, c2, k=1, s=1, p=0)  # common 1x1

        # branches: must take c2 in, produce c_ out
        self.l = Conv(c2, c_, k=3, s=1, p=1)
        self.r = Conv(c2, c_, k=3, s=1, p=1)
        self.m = Conv(c2, c_, k=3, s=1, p=1)

        # DB1: residual bottleneck on c_
        self.db11 = Conv(c_, c_, k=3, s=1, p=1)
        self.db12 = Conv(c_, c_, k=3, s=1, p=1)

        # DB2: residual bottleneck on c_
        self.db21 = Conv(c_, c_, k=3, s=1, p=1)
        self.db22 = Conv(c_, c_, k=3, s=1, p=1)

        # concat 4 maps => 4*c_
        self.out = Conv(4 * c_, c2, k=1, s=1, p=0)

    def forward(self, x):
        x = self.common(x)   # (B, c2, H, W)

        l_out = self.l(x)    # (B, c_, H, W)
        r_out = self.r(x)    # (B, c_, H, W)
        m_out = self.m(x)    # (B, c_, H, W)

        db1_out = m_out + self.db12(self.db11(m_out))         # (B, c_, H, W)
        db2_out = db1_out + self.db22(self.db21(db1_out))     # (B, c_, H, W)

        return self.out(torch.cat([l_out, r_out, db1_out, db2_out], dim=1))

class CSMBNeck1(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        assert c2 % 2 == 0
        c_ = c2 // 2

        # common 1x1 (keep spatial size)
        self.common = Conv(c1, c2, k=1, s=1, p=0)

        # branches: c2 -> c_
        self.l = Conv(c2, c_, k=3, s=1, p=1)
        self.r = Conv(c2, c_, k=3, s=1, p=1)
        self.m = Conv(c2, c_, k=3, s=1, p=1)

        # DB on mid: c_ -> c_ -> c_ with residual add
        self.db1 = Conv(c_, c_, k=3, s=1, p=1)
        self.db2 = Conv(c_, c_, k=3, s=1, p=1)

        # concat 3 maps => 3*c_
        self.out = Conv(3 * c_, c2, k=1, s=1, p=0)

    def forward(self, x):
        x = self.common(x)
        l_out = self.l(x)
        r_out = self.r(x)
        m_out = self.m(x)

        db_out = self.db2(self.db1(m_out)) # no skip connection here

        return self.out(torch.cat([l_out, db_out, r_out], dim=1))
    


# OLD CODE I think this is wrong:

# class CSMBBackBone1(nn.Module):
#     """
#     CSMB with n=1 DB, per Eq.(2)-(3).

#     Two independent conv computations on F:
#       A(F) = Conv3x3(Conv1x1(F))
#       B0(F)= Conv3x3(Conv1x1(F))
#     DB branch:
#       B(F) = DB(B0(F))  (n=1)

#     Concat three feature maps then 1x1:
#       out = Conv1x1( Concat( A(F), B0(F), B(F) ) )
#     """
#     def __init__(self, c1: int, c2: int):
#         super().__init__()
#         assert c2 % 2 == 0, "CSMB uses hidden width c2//2; requires even c2."
#         c_ = c2 // 2

#         # A(F) = Conv3x3(Conv1x1(F))
#         self.a1 = Conv(c1, c_, k=1, s=1, p=0)
#         self.a2 = Conv(c_, c_, k=3, s=1, p=1)

#         # B0(F)= Conv3x3(Conv1x1(F))
#         self.b1 = Conv(c1, c_, k=1, s=1, p=0)
#         self.b2 = Conv(c_, c_, k=3, s=1, p=1)

#         # DB: Conv3x3 -> Conv3x3 + residual
#         self.db1 = Conv(c_, c_, k=3, s=1, p=1)
#         self.db2 = Conv(c_, c_, k=3, s=1, p=1)

#         # Final 1x1 projection after concat (3 * c_ -> c2uja)
#         self.out = Conv(3 * c_, c2, k=1, s=1, p=0)

#     def forward(self, x):
#         a = self.a2(self.a1(x))          # A(F)
#         b0 = self.b2(self.b1(x))         # B0(F)

#         b = b0 + self.db2(self.db1(b0))  # DB(b0), n=1

#         return self.out(torch.cat([a, b0, b], dim=1))



# class CSMBBackBone2(nn.Module):
#     """
#     CSMB with n=2 DB, per Eq.(2)-(3).

#     Same as CSMBBackBone1, but DB is applied twice in series:
#       b1 = DB(b0)
#       b2 = DB(b1)
#     and concat is still exactly three feature maps: [a, b0, b2]
#     """
#     def __init__(self, c1: int, c2: int):
#         super().__init__()
#         assert c2 % 2 == 0, "CSMB uses hidden width c2//2; requires even c2."
#         c_ = c2 // 2

#         # A(F)
#         self.a1 = Conv(c1, c_, k=1, s=1, p=0)
#         self.a2 = Conv(c_, c_, k=3, s=1, p=1)

#         # B0(F)
#         self.b1 = Conv(c1, c_, k=1, s=1, p=0)
#         self.b2 = Conv(c_, c_, k=3, s=1, p=1)

#         # DB #1
#         self.db1_1 = Conv(c_, c_, k=3, s=1, p=1)
#         self.db1_2 = Conv(c_, c_, k=3, s=1, p=1)
#         # DB #2
#         self.db2_1 = Conv(c_, c_, k=3, s=1, p=1)
#         self.db2_2 = Conv(c_, c_, k=3, s=1, p=1)

#         self.out = Conv(3 * c_, c2, k=1, s=1, p=0)

#     def forward(self, x):
#         a = self.a2(self.a1(x))      # A(F)
#         b0 = self.b2(self.b1(x))     # B0(F)

#         b1 = b0 + self.db1_2(self.db1_1(b0))  # DB #1
#         b2 = b1 + self.db2_2(self.db2_1(b1))  # DB #2  (n=2 final)

#         return self.out(torch.cat([a, b0, b2], dim=1))



# class CSMBNeck1(nn.Module):
#     """
#     Same CSMB definition from Eq.(2)-(3), n=1.
#     (Paper uses FPN+PAN in neck; if you still want CSMB in neck, this matches the same formula.)
#     """
#     def __init__(self, c1: int, c2: int):
#         super().__init__()
#         assert c2 % 2 == 0, "CSMB uses hidden width c2//2; requires even c2."
#         c_ = c2 // 2

#         self.a1 = Conv(c1, c_, k=1, s=1, p=0)
#         self.a2 = Conv(c_, c_, k=3, s=1, p=1)

#         self.b1 = Conv(c1, c_, k=1, s=1, p=0)
#         self.b2 = Conv(c_, c_, k=3, s=1, p=1)

#         self.db1 = Conv(c_, c_, k=3, s=1, p=1)
#         self.db2 = Conv(c_, c_, k=3, s=1, p=1)

#         self.out = Conv(3 * c_, c2, k=1, s=1, p=0)

#     def forward(self, x):
#         a = self.a2(self.a1(x))
#         b0 = self.b2(self.b1(x))
#         b = b0 + self.db2(self.db1(b0))
#         return self.out(torch.cat([a, b0, b], dim=1))




class DWConvModule(nn.Module):
    """
    Middle LKSP branch:
      DWConv(1×K, dilation=K) -> DWConv(K×1, dilation=K) -> PWConv(1×1)
    (Conv-only inside branch, no BN/act)
    """
    def __init__(self, c: int, k: int):
        super().__init__()
        d = k  # paper: dilation = K
        p = (k - 1) * d // 2  # "same" pad for stride=1 with dilation

        # 1×K depthwise (dilate along width)
        self.dw1 = nn.Conv2d(
            c, c, kernel_size=(1, k), stride=1,
            padding=(0, p), dilation=(1, d),
            groups=c, bias=True
        )
        # K×1 depthwise (dilate along height)
        self.dw2 = nn.Conv2d(
            c, c, kernel_size=(k, 1), stride=1,
            padding=(p, 0), dilation=(d, 1),
            groups=c, bias=True
        )
        # pointwise 1×1
        self.pw = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.dw1(x)  # 1*K first
        x = self.dw2(x)  # then K*1
        return self.pw(x) # then normal pointwise conv


class LKSP(nn.Module):
    """
    LKSP with common conv keeping c_out:

    Common:
      x -> common Conv1×1 (c1 -> c_out), Conv-only

    Branch inputs:
      - skip + 3 DW branches share: reduce Conv1×1 (c_out -> 0.5*c_out), Conv-only
      - GAP branch uses common output: AAP -> Conv1×1 (to 0.5*c_out) -> ReLU -> Upsample

    Concat 5 branches (each 0.5*c_out => 2.5*c_out) then OutputConv:
      Conv -> BN -> ReLU -> Dropout
    """
    def __init__(self, c1: int, c2: int = None, drop: float = 0.1):
        super().__init__()
        c2 = c1 if c2 is None else c2
        assert c2 % 2 == 0, "LKSP expects even c_out."
        assert 0.0 <= drop <= 1.0, f"drop must be in [0,1], got {drop}"

        mid = c2 // 2

        # Common conv: Conv ONLY, keeps channels at c_out
        self.common = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=True)

        # Shared reduce for skip + DW branches: Conv ONLY to 0.5*c_out
        self.reduce = nn.Conv2d(c2, mid, kernel_size=1, stride=1, padding=0, bias=True)

        # Three DW branches on reduced feature
        self.m4 = DWConvModule(mid, k=4)
        self.m5 = DWConvModule(mid, k=5)
        self.m6 = DWConvModule(mid, k=6)

        # GAP branch on common feature: AAP -> Conv -> ReLU -> upsample
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.gap_conv = nn.Conv2d(c2, mid, kernel_size=1, stride=1, padding=0, bias=True)
        self.gap_act = nn.ReLU(inplace=True)

        # OutputConv: Conv -> BN -> ReLU -> Dropout
        # concat channels = 5 * mid = 2.5 * c_out
        self.out = nn.Sequential(
            nn.Conv2d(5 * mid, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop),
        )

    def forward(self, x):
        x0 = self.common(x)          # (B, c_out, H, W)

        # shared reduced feature for skip + DW branches
        t = self.reduce(x0)          # (B, mid, H, W)
        H, W = t.shape[-2:]

        skip = t
        b1 = self.m4(t)
        b2 = self.m5(t)
        b3 = self.m6(t)

        # GAP branch uses the common conv output x0
        g = self.aap(x0)                         # (B, c_out, 1, 1)
        g = self.gap_act(self.gap_conv(g))       # (B, mid, 1, 1)
        g = F.interpolate(g, size=(H, W), mode="nearest")

        y = torch.cat([skip, b1, b2, b3, g], dim=1)  # (B, 5*mid, H, W)
        return self.out(y)



  

# ---------------------------------------------------------------------------
# AquaYOLO custom blocks 
# ---------------------------------------------------------------------------

class AquaResidualBlock(nn.Module):
    """
    AquaResidualBlock (paper-style residual block) with a projection shortcut when needed.

    What the paper shows (Figure 1):
        main path: 3x3 Conv -> ReLU -> 3x3 Conv
        skip path: x (identity)
        output: ReLU( main + skip )

    The catch (your backbone diagram):
        Some of these "ResNet" blocks use stride=2 to downsample spatially and/or change channels.
        If the main path changes shape (H,W,C), then the skip path can't be raw x anymore,
        because you cannot add tensors of different shapes.

    So we do the standard ResNet trick:
        - If shapes match: skip = x (identity)
        - If shapes differ: skip = 1x1 Conv(x) with stride=s to match (H,W,C)

    No BatchNorm inside this block (as described in the paper text).
    """

    def __init__(
        self,
        c1: int,           # input channels
        c2: int,           # output channels
        s: int = 1,        # stride (s=2 does downsampling)
        k: int = 3,        # kernel size for the 3x3 convs (paper uses 3)
        p: int = None,     # padding; if None we use "same" padding for odd kernels
        bias: bool = True  # bias=True is reasonable since there's no BN inside the block
    ):
        super().__init__()

        # For k=3, "same" padding is p=1 so spatial size is preserved when stride=1.
        if p is None:
            p = k // 2

        # Main branch: g(x)
        # conv1: 3x3, stride = s
        # If s=2, this is where we reduce H,W by half.
        self.conv1 = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=bias
        )
        self.relu1 = nn.ReLU(inplace=True)

        # conv2: 3x3, stride = 1
        # Keeps the already-downsampled size produced by conv1.
        self.conv2 = nn.Conv2d(
            in_channels=c2,
            out_channels=c2,
            kernel_size=k,
            stride=1,
            padding=p,
            bias=bias
        )

        # Skip branch: x (identity) OR projection
        # We need skip to have the SAME SHAPE as g(x) so we can add:
        #   y = g(x) + skip
        #
        # g(x) has shape: [B, c2, H/s, W/s]  (s is 1 or 2)
        # raw x has shape: [B, c1, H,   W]
        #
        # If (c1==c2 and s==1), shapes match and we can use identity.
        # Otherwise, we use a 1x1 conv with stride s to match both channels AND spatial size.
        if (c1 == c2) and (s == 1):
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                in_channels=c1,
                out_channels=c2,
                kernel_size=1,   # 1x1 projection
                stride=s,        # stride must match conv1 so H,W match
                padding=0,
                bias=bias
            )

        # Final activation after the addition (matches Figure 1)
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward:
          1) compute main = g(x) = conv2(ReLU(conv1(x)))
          2) compute skip = x      (identity) OR 1x1 projection
          3) add: out = main + skip
          4) ReLU after the addition
        """

        # ---- main path g(x) ----
        main = self.conv1(x)
        main = self.relu1(main)
        main = self.conv2(main)

        # ---- skip path ----
        skip = self.skip(x)

        # ---- residual add + activation ----
        out = main + skip
        out = self.relu_out(out)
        return out


# Basic Conv + BN + Activation 
class AQUAYOLO_ConvBNAct(nn.Module):
    """
    Small helper block to match the paper diagrams:
      Conv -> BatchNorm -> ReLU

    We use bias=False because BN has its own affine parameters.
    """
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2  # "same" padding for odd kernels
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvOnly(nn.Module):
    """
    Pure conv with no BN/activation.
    The paper diagrams show some conv stacks without explicit BN/activation blocks (purple boxes).
    """
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, bias: bool = True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FAU(nn.Module):
    """
    Feature Alignment Unit (FAU) as described in the paper:
      (Up/Down Sampling) + Conv + BN + ReLU 

    Purpose:
      Take a feature map from some level (Fb) and align it to the "current level" (Fa)
      so they can be combined.
      - Spatial alignment: resize to (H_a, W_a)
      - Channel alignment: project channels to c_out (normally channels of Fa)
    """
    def __init__(self, c_in: int, c_out: int, k: int = 1):
        super().__init__()
        # Use k=1 by default: cheap channel alignment after resizing.
        # You can set k=3 if you want more spatial mixing inside FAU.
        self.proj = AQUAYOLO_ConvBNAct(c_in, c_out, k=k, s=1)

    def forward(self, x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        # 1) spatial alignment (upsample OR downsample)
        if x.shape[-2:] != target_hw:
            # 'nearest' matches typical YOLO neck behavior and avoids smoothing edges
            x = F.interpolate(x, size=target_hw, mode="nearest")

        # 2) channel alignment + nonlinearity (Conv+BN+ReLU)
        return self.proj(x)


class CAFS(nn.Module):
    """
    CAFS: Context-Aware Feature Selection

    Inputs:
      Fa, Fb: feature maps that should already be aligned to the same spatial size (H,W).
              Channels can be assumed the same too in the typical DSAM usage.

    Diagram logic (Figure 3):
      1) Fa -> ConvBNReLU
         Fb -> ConvBNReLU
      2) concat -> trunk ConvBNReLU -> trunk ConvBNReLU   (context feature)
      3) Wf branch:
           trunk -> Conv(k=1) -> ConvBNReLU -> Sigmoid -> multiply with trunk -> Wf
      4) Wa/Wb branch:
           trunk -> Conv(k=1) -> Conv(k=3) -> Conv(k=3) -> ConvBNReLU -> Softmax -> Wa,Wb
      5) selection:
           Wa * Fa + Wb * Fb
      6) output:
           (Wa*Fa + Wb*Fb) + Wf

    Notes:
      - Wa/Wb are *spatial* weights (shape Bx1xHxW each).
      - Wf is a *feature map* (shape BxCxHxW).
      - This module does NOT change spatial resolution.
      - Output channels = c (the working channel width).
    """

    def __init__(self, c: int, do_pre: bool = True, hidden_ratio: float = 0.25):
        super().__init__()
        self.c = c
        self.do_pre = do_pre

        # -------------------------------------------------------
        # (A) Pre-processing before concat (top two yellow boxes)
        # -------------------------------------------------------
        # These are applied separately to Fa and Fb BEFORE concatenation.
        # If your DSAM already did these convs, set do_pre=False. IT DOES NOT
        if do_pre:
            self.pre_a = AQUAYOLO_ConvBNAct(c, c, k=3, s=1)
            self.pre_b = AQUAYOLO_ConvBNAct(c, c, k=3, s=1)
        else:
            self.pre_a = nn.Identity()
            self.pre_b = nn.Identity()

        # -------------------------------------------------------
        # (B) Trunk after concat (two middle yellow boxes)
        # -------------------------------------------------------
        # Concat makes channels 2c, then we compress back to c.
        self.trunk1 = AQUAYOLO_ConvBNAct(2 * c, c, k=3, s=1)
        self.trunk2 = AQUAYOLO_ConvBNAct(c, c, k=3, s=1)

        # -------------------------------------------------------
        # (C) Wf branch (left side in the figure)
        #     trunk -> Conv(k=1) -> ConvBNReLU -> Sigmoid -> multiply -> Wf
        # -------------------------------------------------------
        self.wf_conv1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)  # purple Conv(k=1)
        self.wf_conv2 = AQUAYOLO_ConvBNAct(c, c, k=3, s=1)                                      # yellow Conv+BN+ReLU
        self.wf_sigmoid = nn.Sigmoid()

        # -------------------------------------------------------
        # (D) Wa/Wb branch (right side in the figure)
        #     trunk -> Conv(k=1) -> Conv(k=3) -> Conv(k=3) -> ConvBNReLU -> Softmax -> Wa,Wb
        # -------------------------------------------------------
        hidden = max(8, int(c * hidden_ratio))

        self.wab_conv1 = nn.Conv2d(c, hidden, kernel_size=1, stride=1, padding=0, bias=True)  # purple Conv(k=1)
        self.wab_conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=True)  # purple Conv(k=3)
        self.wab_conv3 = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=True)  # purple Conv(k=3)
        self.wab_post  = AQUAYOLO_ConvBNAct(hidden, hidden, k=3, s=1)  # yellow Conv+BN+ReLU

        # Produce 2 logits maps (Wa_logit and Wb_logit), then softmax across the 2 channels
        self.wab_logits = nn.Conv2d(hidden, 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        # Sanity: CAFS expects both feature maps already aligned in size.
        assert fa.shape[-2:] == fb.shape[-2:], f"Fa/Fb spatial mismatch: {fa.shape} vs {fb.shape}"

        # If channels differ, you should align them BEFORE calling CAFS (typically via FAU).
        assert fa.shape[1] == self.c and fb.shape[1] == self.c, \
            f"CAFS expects channel={self.c}, got Fa={fa.shape[1]}, Fb={fb.shape[1]}"

        # -------------------------
        # 1) Pre-conv on Fa and Fb
        # -------------------------
        fa0 = self.pre_a(fa)  # ConvBNReLU(Fa) in the diagram
        fb0 = self.pre_b(fb)  # ConvBNReLU(Fb)

        # -------------------------
        # 2) Concat and trunk convs
        # -------------------------
        x = torch.cat([fa0, fb0], dim=1)   # Concatenation block
        x = self.trunk1(x)                # trunk conv 1
        x = self.trunk2(x)                # trunk conv 2
        # x is now the "context feature" used by both branches.

        # -------------------------
        # 3) Wf branch (sigmoid)
        # -------------------------
        # gate in [0,1]
        wf_gate = self.wf_sigmoid(self.wf_conv2(self.wf_conv1(x)))
        # element-wise multiplication (purple ⊗ in the figure)
        wf = wf_gate * x

        # -------------------------
        # 4) Wa/Wb branch (softmax)
        # -------------------------
        wab = self.wab_conv1(x)
        wab = self.wab_conv2(wab)
        wab = self.wab_conv3(wab)
        wab = self.wab_post(wab)

        wab_logits = self.wab_logits(wab)     # (B,2,H,W)
        wab = self.softmax(wab_logits)        # softmax over the 2 channels => Wa/Wb

        wa = wab[:, 0:1, :, :]                # (B,1,H,W)
        wb = wab[:, 1:2, :, :]                # (B,1,H,W)

        # -------------------------
        # 5) Apply Wa/Wb to Fa and Fb
        # -------------------------
        # The figure shows Wa and Wb multiplying Fa and Fb (purple ⊗), then summed (green ⊕).
        selected = wa * fa + wb * fb

        # -------------------------
        # 6) Final output = selected + Wf
        # -------------------------
        out = selected + wf
        return out


class DSAM(nn.Module):
    """
    DSAM replicating the *two-path* structure in Figure 2:

    LEFT (interaction) PATH:
      Fa -> FAU
      Fb -> FAU
      element-wise multiplication (⊗)  -> produces interaction feature

    RIGHT (selection) PATH:
      Fa -> ConvBNReLU -> FAU -> ConvBNReLU
      Fb -> ConvBNReLU -> FAU -> ConvBNReLU
      -> CAFS -> ConvBNReLU

    Then the diagram shows the interaction result being combined with the right-path output
    (element-wise addition) and a 1x1 conv appearing on that combined path.

    Output channels:
      By default, output channels = channels(Fa) (paper-faithful; FAU aligns to Fa). 
      Optionally, you can pass c_out to force a fixed width.
    """
    def __init__(
        self,
        ch_in: Sequence[int],        # [c_fa, c_fb] (parse_model passes this)
        c_out: Optional[int] = None, # optional override
        fau_k: int = 1,              # 1x1 alignment conv inside FAU
    ):
        super().__init__()
        assert len(ch_in) == 2, "DSAM expects ch_in=[c_fa, c_fb]"
        c_fa, c_fb = int(ch_in[0]), int(ch_in[1])

        # Choose output width.
        # Paper intent: align to current level Fa => c_out = c_fa by default.
        self.c_out = c_fa if c_out is None else int(c_out)

        # ---------- LEFT path (interaction) ----------
        # FAU aligns both features to (H_a, W_a, c_out) so we can multiply them.
        self.fau_left_a = FAU(c_fa, self.c_out, k=fau_k)
        self.fau_left_b = FAU(c_fb, self.c_out, k=fau_k)

        # After we combine (multiply + add later), the figure shows a Conv(k=1) block.
        self.left_conv1x1 = nn.Conv2d(self.c_out, self.c_out, kernel_size=1, stride=1, padding=0, bias=True)

        # ---------- RIGHT path (selection + CAFS) ----------
        # Each input goes through Conv+BN+ReLU BEFORE FAU in the figure.
        self.pre_a = AQUAYOLO_ConvBNAct(c_fa, self.c_out, k=3, s=1)
        self.pre_b = AQUAYOLO_ConvBNAct(c_fb, self.c_out, k=3, s=1)

        # Then FAU blocks (align spatially to Fa's size, keep c_out channels)
        self.fau_right_a = FAU(self.c_out, self.c_out, k=fau_k)
        self.fau_right_b = FAU(self.c_out, self.c_out, k=fau_k)

        # Then Conv+BN+ReLU AFTER FAU in the figure
        self.post_a = AQUAYOLO_ConvBNAct(self.c_out, self.c_out, k=3, s=1)
        self.post_b = AQUAYOLO_ConvBNAct(self.c_out, self.c_out, k=3, s=1)

        # CAFS selection/fusion + final Conv+BN+ReLU
        self.cafs = CAFS(self.c_out)
        self.right_out = AQUAYOLO_ConvBNAct(self.c_out, self.c_out, k=3, s=1)

    def forward(self, x):
        """
        Ultralytics will pass x as [Fa, Fb] because your YAML uses from=[[...], ...]
        """
        assert isinstance(x, (list, tuple)) and len(x) == 2, "DSAM forward expects [Fa, Fb]"
        fa, fb = x

        # DSAM is defined relative to Fa (current level).
        target_hw = fa.shape[-2:]

        # =========================
        # LEFT: interaction branch
        # =========================
        # 1) align both features to Fa resolution + c_out channels
        la = self.fau_left_a(fa, target_hw)  # (B, c_out, H_a, W_a)
        lb = self.fau_left_b(fb, target_hw)  # (B, c_out, H_a, W_a)

        # 2) element-wise multiplication (purple ⊗ in the figure)
        inter = la * lb                      # (B, c_out, H_a, W_a)

        # =========================
        # RIGHT: CAFS branch
        # =========================
        # 1) Conv+BN+ReLU before FAU
        ra = self.pre_a(fa)
        rb = self.pre_b(fb)

        # 2) FAU alignment (spatial) to Fa size
        ra = self.fau_right_a(ra, target_hw)
        rb = self.fau_right_b(rb, target_hw)

        # 3) Conv+BN+ReLU after FAU
        ra = self.post_a(ra)
        rb = self.post_b(rb)

        # 4) CAFS + final Conv+BN+ReLU
        cafs_out = self.cafs(ra, rb)
        right = self.right_out(cafs_out)

        # =========================
        # Combine branches (matches the add node + Conv(k=1) in the figure)
        # =========================
        # The diagram shows an element-wise addition node that mixes the interaction result with the CAFS path.
        # Then a 1x1 conv appears on that combined path.
        fused = right + inter
        fused = self.left_conv1x1(fused)

        return fused





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




# ==============================================================
# yolov11-SDC
# ===============================================================

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


from ..backbone.UniRepLKNet import get_bn, get_conv2d, NCHWtoNHWC, GRNwithNHWC, SEBlock, NHWCtoNCHW, fuse_bn, merge_dilated_into_large_kernel
from timm.models.layers import DropPath
class DilatedReparamBlock(nn.Module):

    def __init__(self, channels, kernel_size, deploy=False, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def switch_to_deploy(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class UniRepLKNetBlock(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 deploy=False,
                 attempt_use_lk_impl=True,
                 with_cp=False,
                 use_sync_bn=False,
                 ffn_factor=4):
        super().__init__()
        self.with_cp = with_cp

        self.need_contiguous = (not deploy) or kernel_size >= 7

        if kernel_size == 0:
            self.dwconv = nn.Identity()
            self.norm = nn.Identity()
        elif deploy:
            self.dwconv = get_conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=True,
                                     attempt_use_lk_impl=attempt_use_lk_impl)
            self.norm = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              use_sync_bn=use_sync_bn,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
        elif kernel_size == 1:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=1, bias=deploy)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
        else:
            assert kernel_size in [3, 5]
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=dim, bias=deploy)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)

        self.se = SEBlock(dim, dim // 4)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim),
                NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                get_bn(dim, use_sync_bn=use_sync_bn))

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if (not deploy) and layer_scale_init_value is not None \
                                                         and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, inputs):

        def _f(x):
            if self.need_contiguous:
                x = x.contiguous()
            y = self.se(self.norm(self.dwconv(x)))
            y = self.pwconv2(self.act(self.pwconv1(y)))
            if self.gamma is not None:
                y = self.gamma.view(1, -1, 1, 1) * y
            return self.drop_path(y) + x

        if self.with_cp and inputs.requires_grad:
            return checkpoint.checkpoint(_f, inputs)
        else:
            return _f(inputs)

    def switch_to_deploy(self):
        if hasattr(self.dwconv, 'switch_to_deploy'):
            self.dwconv.switch_to_deploy()
        if hasattr(self.norm, 'running_var') and hasattr(self.dwconv, 'lk_origin'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
            self.dwconv.lk_origin.bias.data = self.norm.bias + (self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            self.norm = nn.Identity()
        if self.gamma is not None:
            final_scale = self.gamma.data
            self.gamma = None
        else:
            final_scale = 1
        if self.act[1].use_bias and len(self.pwconv2) == 3:
            grn_bias = self.act[1].beta.data
            self.act[1].__delattr__('beta')
            self.act[1].use_bias = False
            linear = self.pwconv2[0]
            grn_bias_projected_bias = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
            bn = self.pwconv2[2]
            std = (bn.running_var + bn.eps).sqrt()
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = linear.weight * (bn.weight / std * final_scale).view(-1, 1)
            linear_bias = 0 if linear.bias is None else linear.bias.data
            linear_bias += grn_bias_projected_bias
            new_linear.bias.data = (bn.bias + (linear_bias - bn.running_mean) * bn.weight / std) * final_scale
            self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])

class C3k_UniRepLKNetBlock(C3k):
    def __init__(self, c1, c2, n=1, k=7, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(UniRepLKNetBlock(c_, k) for _ in range(n)))

class C3k2_UniRepLKNetBlock(C3k2):
    def __init__(self, c1, c2, n=1, k=7, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_UniRepLKNetBlock(self.c, self.c, 2, k, shortcut, g) if c3k else UniRepLKNetBlock(self.c, k) for _ in range(n))

class Bottleneck_DRB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DilatedReparamBlock(c2, 7)

class C3k_DRB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DRB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_DRB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DRB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DRB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))



class SF(nn.Module):
    def __init__(self, channel=512, features_to_keep=32, *args, **kwargs):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.features_to_keep = features_to_keep

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        mean_y = torch.mean(y, dim=0, keepdim=True)
        _, indices = torch.topk(mean_y, min(self.features_to_keep, c), dim=1)
        indices = indices.repeat(b, 1)
        reduced_features = torch.gather(x, 1, indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w))

        return reduced_features


class SpatialAttention_CGA(nn.Module):
    def __init__(self):
        super(SpatialAttention_CGA, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention_CGA(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention_CGA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention_CGA(nn.Module):
    def __init__(self, dim):
        super(PixelAttention_CGA, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)
        pattn1 = pattn1.unsqueeze(dim=2)
        x2 = torch.cat([x, pattn1], dim=2)
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(dim, reduction)
        self.pa = PixelAttention_CGA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, y = data
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result






