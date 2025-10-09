import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- GSConv ----------
class GSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        c_mid = max(1, c2 // 2)
        self.sc = nn.Conv2d(c1, c_mid, k, s, p, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.dw = nn.Conv2d(c_mid, c_mid, k, 1, p, groups=c_mid, bias=False)
        self.pw = nn.Conv2d(c_mid, c_mid, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.act = nn.SiLU() if act else nn.Identity()
        self.out_pw = nn.Conv2d(c_mid, c2, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)

    def channel_shuffle(self, x, groups=2):
        b, c, h, w = x.size()
        g = min(groups, c)
        assert c % g == 0
        cp = c // g
        x = x.view(b, g, cp, h, w).transpose(1, 2).contiguous().view(b, c, h, w)
        return x

    def forward(self, x):
        x = self.act(self.bn1(self.sc(x)))
        x = self.dw(x); x = self.pw(x)
        x = self.act(self.bn2(x))
        x = self.channel_shuffle(x)
        x = self.bn3(self.out_pw(x))
        return self.act(x)

# ---------- FasterNet blocks ----------
class PConv(nn.Module):
    def __init__(self, c, r=0.25, k=3, s=1):
        super().__init__()
        p = k // 2
        cp = max(1, int(c * r))
        self.cp = cp
        self.conv = nn.Conv2d(cp, cp, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(cp)
        self.act = nn.GELU()

    def forward(self, x):
        x1, x2 = torch.split(x, [self.cp, x.shape[1]-self.cp], dim=1)
        y1 = self.act(self.bn(self.conv(x1)))
        return torch.cat([y1, x2], dim=1)

class FasterBlock(nn.Module):
    def __init__(self, c, r=0.25):
        super().__init__()
        self.pconv = PConv(c, r=r)
        self.expand = nn.Sequential(
            nn.Conv2d(c, c*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c*2), nn.GELU(),
            nn.Conv2d(c*2, c, 1, 1, 0, bias=False)
        )

    def forward(self, x): return self.expand(self.pconv(x))

# -------- LC2f: C2f with FasterBlock + GSConv (Fig. 4) --------
class LC2f(nn.Module):
    # match Ultralytics C2f signature
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # internal expansion like C2f (if you want to mimic)
        self.cv1 = nn.Sequential(nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
                                 nn.BatchNorm2d(c2), nn.SiLU())
        self.m = nn.Sequential(*[FasterBlock(c2) for _ in range(max(1, n))])
        self.cv2 = GSConv(c2, c2)
        self.add = shortcut
    def forward(self, x):
        y = self.cv1(x)
        y = self.m(y)
        y = self.cv2(y)
        return y + x if self.add and y.shape == x.shape else y



