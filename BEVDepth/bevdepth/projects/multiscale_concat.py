import torch as th
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConcat(nn.Module):
    def __init__(self, in_chs=[256,512,1024,1024], out_dim=256, mid=256):
        super().__init__()
        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(in_chs), 2*mid, kernel_size=1, bias=False),
            nn.GroupNorm(32, 2*mid),           
            nn.SiLU(inplace=True),

            nn.Conv2d(2*mid, 2*mid, kernel_size=3, padding=1, groups=2*mid, bias=False),
            nn.GroupNorm(32, 2*mid),
            nn.SiLU(inplace=True),

            nn.Conv2d(2*mid, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, out_dim),       
        )

    def forward(self, xs):
        x0, x1, x2, x3 = xs
        B, _, H, W = x0.shape
        B, _, H1, W1 = x1.shape
                   
        f1 = th.cat([x2, x3], dim=1)     # H//4, W//4
        f2 = th.cat([x1, F.interpolate(f1, (H1, W1), mode='bilinear', align_corners=False)], dim=1)     # H//2, W//2
        x = th.cat([x0, F.interpolate(f2, (H, W), mode='bilinear', align_corners=False)], dim=1)     # H, W  
        return self.out_layer(x)           # [B,out_dim,H,W]
    
    
class MultiScaleConcatV2(nn.Module):
    def __init__(self, in_chs=[256,512,1024,1024], out_dim=256, mid=256):
        """
        Hierarchical feature fusion by simple concatenation and conv
        """
        super().__init__()
        
        C0, C1, C2, C3 = in_chs

        self.p2 = nn.Sequential(nn.Conv2d(C2, C2, 1, bias=False), nn.GroupNorm(32, C2))
        self.p3 = nn.Sequential(nn.Conv2d(C3, C3, 1, bias=False), nn.GroupNorm(32, C3))
        
        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(in_chs), mid, kernel_size=1, bias=False),
            nn.GroupNorm(32, mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, out_dim, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, xs):
        x0, x1, x2, x3 = xs
        B, _, H, W = x0.shape     # H, W
        B, _, H1, W1 = x1.shape   # H//2, W//2
        
        x2 = self.p2(x2)
        x3 = self.p3(x3)

        f1 = th.cat([x2, x3], dim=1)     # H//4, W//4
        f2 = th.cat([x1, F.interpolate(f1, (H1, W1), mode='bilinear', align_corners=False)], dim=1)     # H//2, W//2
        x = th.cat([x0, F.interpolate(f2, (H, W), mode='bilinear', align_corners=False)], dim=1)     # H, W  
        return self.out_layer(x)           # [B,out_dim,H,W]


class MultiScaleConcatV3(nn.Module):
    def __init__(self, in_chs=[256,512,1024,1024], out_dim=256, mid=256):
        """
        Hierarchical feature fusion by simple concatenation and conv
        """
        super().__init__()
        
        C0, C1, C2, C3 = in_chs
        d0, d1, d2, d3 = 256, 128, 128, 128
        
        self.x0_clean = nn.Sequential(
            nn.GroupNorm(32, C0),
            nn.SiLU(inplace=True),
            nn.Conv2d(C0, C0, 3, padding=1, groups=C0, bias=False),
        )
        self.gamma0 = nn.Parameter(th.tensor(1e-3))
        
        self.p1 = nn.Conv2d(512, d1, 1, bias=False)
        self.p2 = nn.Sequential(nn.Conv2d(C2, d2, 1, bias=False), nn.GroupNorm(16, d2))
        self.p3 = nn.Sequential(nn.Conv2d(C3, d3, 1, bias=False), nn.GroupNorm(16, d3))
        
        in_sum = d0 + d1 + d2 + d3  
        
        self.out_layer = nn.Sequential(
            nn.Conv2d(in_sum, mid, kernel_size=1, bias=False),
            nn.GroupNorm(32, mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, out_dim, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, xs):
        x0, x1, x2, 
        B, _, H, W = x0.shape     # H, W
        B, _, H1, W1 = x1.shape   # H//2, W//2
        
        x0 = x0 + self.gamma0 * self.x0_clean(x0)
        x1 = self.p1(x1)
        x2 = self.p2(x2)
        x3 = self.p3(x3)

        f1 = th.cat([x2, x3], dim=1)     # H//4, W//4
        f2 = th.cat([x1, F.interpolate(f1, (H1, W1), mode='bilinear', align_corners=False)], dim=1)     # H//2, W//2
        x = th.cat([x0, F.interpolate(f2, (H, W), mode='bilinear', align_corners=False)], dim=1)     # H, W  
        return self.out_layer(x)           # [B,out_dim,H,W]


class MultiScaleConcatFusion(nn.Module):
    """
    Scale-aware fusion: (objectness)
      - x0: object-focused (local) → higher weight at regions with high objectness
      - x1: mid-scale representation
      - x2, x3: global context → higher weight at background regions (low objectness)
    """
    def __init__(self, in_chs=[256, 512, 1024, 1024], out_dim=256, mid=256, T=1.0, beta=2.0):
        super().__init__()
        C0, C1, C2, C3 = in_chs
        self.T = T         # Softmax temperature (smaller → sharper distribution)
        self.beta = beta   # Strength of the objectness modulation

        # Channel projection for each scale to a common reduced dimension 
        d0, d1, d2, d3 = 256, 192, 128, 128
        self.p0 = nn.Conv2d(C0, d0, 1, bias=False)
        self.p1 = nn.Conv2d(C1, d1, 1, bias=False)
        # self.p2 = nn.Conv2d(C2, d2, 1, bias=False)
        # self.p3 = nn.Conv2d(C3, d3, 1, bias=False)
        self.p2 = nn.Sequential(
            nn.Conv2d(C2, d2, 1, bias=False),
            nn.GroupNorm(32, d2),      # normalization only for global context
            # nn.InstanceNorm2d(d2, affine=True)
            )
        self.p3 = nn.Sequential(
            nn.Conv2d(C3, d3, 1, bias=False),
            nn.GroupNorm(32, d3),
            # nn.InstanceNorm2d(d3, affine=True)
            )

        # Logit maps (1 channel) for spatial scale gating
        self.l0 = nn.Conv2d(d0, 1, 1)
        self.l1 = nn.Conv2d(d1, 1, 1)
        self.l2 = nn.Conv2d(d2, 1, 1)
        self.l3 = nn.Conv2d(d3, 1, 1)

        # Objectness map computed from x0 (highest-resolution feature)
        self.obj_head = nn.Sequential(
            nn.Conv2d(d0, d0, 3, padding=1, groups=d0, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(d0, 1, 1)
        )

        # Global scalar gates for each scale (also act as initial priors)
        self.alpha0 = nn.Parameter(th.tensor(1.25))  # x0 prior ↑ (object-centric)
        self.alpha1 = nn.Parameter(th.tensor(1.00))  # x1 prior (mid-scale)
        self.alpha2 = nn.Parameter(th.tensor(0.90))  # x2 prior (context)
        self.alpha3 = nn.Parameter(th.tensor(0.90))  # x3 prior (context)

        # Final fusion head: shallow block (normalization applied only at the front)
        in_cat = d0 + d1 + d2 + d3
        self.out_head = nn.Sequential(
            nn.Conv2d(in_cat, mid, 1, bias=False),
            nn.GroupNorm(32, mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),  # depthwise conv
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, out_dim, 1, bias=False),
        )

        # Balanced initialization: initialize all gating logits with zero bias
        for g in [self.l0, self.l1, self.l2, self.l3]:
            nn.init.zeros_(g.bias)

    def forward(self, xs):
        """
        Expected input resolutions:
          x0: (B,256, H,  W)   ← object-focused, highest resolution
          x1: (B,512, H/2,W/2) ← mid-level
          x2: (B,1024,H/4,W/4) ← global
          x3: (B,1024,H/4,W/4) ← global
        """
        x0, x1, x2, x3 = xs
        B, _, H, W = x0.shape

        # 1) Channel alignment (per-scale projection)
        f0 = self.p0(x0)                                             # (B,d0,H,W)
        f1 = F.interpolate(self.p1(x1), (H, W), mode='bilinear', align_corners=False)  # (B,d1,H,W)
        f2 = F.interpolate(self.p2(x2), (H, W), mode='bilinear', align_corners=False)  # (B,d2,H,W)
        f3 = F.interpolate(self.p3(x3), (H, W), mode='bilinear', align_corners=False)  # (B,d3,H,W)

        # 2) Compute spatial objectness map from x0
        #    Sigmoid output in [0,1]; higher around objects, lower in background
        obj = th.sigmoid(self.obj_head(f0))                          # (B,1,H,W)

        # 3) Compute scale-wise logits and apply semantic bias via objectness
        #    - Near objects: add +β·obj to x0/x1 logits
        #    - Background:  add +β·(1-obj) to x2/x3 logits
        l0 = self.l0(f0) + self.beta * obj
        l1 = self.l1(f1) + 0.6 * self.beta * obj                     # mid-scale receives weaker bias
        back = (1.0 - obj)
        l2 = self.l2(f2) + self.beta * back
        l3 = self.l3(f3) + self.beta * back

        # 4) Normalize scale weights using softmax (across scale dimension)
        logits = th.cat([l0, l1, l2, l3], dim=1) / self.T            # (B,4,H,W)
        w = th.softmax(logits, dim=1)
        w0, w1, w2, w3 = w[:, 0:1], w[:, 1:2], w[:, 2:3], w[:, 3:4]

        # 5) Apply global scalar gates and element-wise weighting
        f0w = self.alpha0 * w0 * f0
        f1w = self.alpha1 * w1 * f1
        f2w = self.alpha2 * w2 * f2
        f3w = self.alpha3 * w3 * f3

        # 6) Concatenate weighted features and pass through the shallow fusion head
        z = th.cat([f0w, f1w, f2w, f3w], dim=1)
        out = self.out_head(z)
        return out, {'weights': w, 'obj': obj}



class MultiScaleConcatWeighted(nn.Module):
    def __init__(self, in_chs=[256, 512, 1024, 1024], out_dim=256, mid=256, T=1.0):
        """
        Learned weighted fusion at each merge step (x2<->x3, x1<->up(f1), x0<->up(f2)).
        Weights are spatial (H×W), sum-to-one via softmax (temperature T).
        """
        super().__init__()
        C0, C1, C2, C3 = in_chs
        self.T = T

        # Gating (branch logits: 1ch)
        self.g2  = nn.Conv2d(C2, 1, kernel_size=1)
        self.g3  = nn.Conv2d(C3, 1, kernel_size=1)
        self.g1  = nn.Conv2d(C1, 1, kernel_size=1)
        self.gf1 = nn.Conv2d(C2 + C3, 1, kernel_size=1)
        self.g0  = nn.Conv2d(C0, 1, kernel_size=1)
        self.gf2 = nn.Conv2d(C1 + C2 + C3, 1, kernel_size=1)

        # Global scalar gates (optional fine-tuning of branch dominance)
        self.alpha2  = nn.Parameter(th.tensor(1.0))
        self.alpha3  = nn.Parameter(th.tensor(1.0))  # mid-block 
        self.alpha1  = nn.Parameter(th.tensor(1.0))
        self.alphaf1 = nn.Parameter(th.tensor(1.0))
        self.alpha0  = nn.Parameter(th.tensor(2.0))
        self.alphaf2 = nn.Parameter(th.tensor(1.0))

        # Output head (kept as-is)
        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(in_chs), mid, kernel_size=1, bias=False),   # 2816→256 
            nn.GroupNorm(32, mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),  # depthwise
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, out_dim, kernel_size=1, bias=False),      
        )
        # Balanced init: start with near-equal weights
        nn.init.zeros_(self.g2.bias);  nn.init.zeros_(self.g3.bias)
        nn.init.zeros_(self.g1.bias);  nn.init.zeros_(self.gf1.bias)
        nn.init.zeros_(self.g0.bias);  nn.init.zeros_(self.gf2.bias)

    def forward(self, xs):
        x0, x1, x2, x3 = xs       # out 50, out 25, out 12, mid 12
        B, _, H,  W  = x0.shape      # 50x50
        _, _, H1, W1 = x1.shape      # 25x25 
        H2, W2       = x2.shape[-2:] # 12x12

        # ---------- Stage A: x2 ⟷ (x3→H/4) ----------
        x3_up = F.interpolate(x3, (H2, W2), mode='bilinear', align_corners=False)
        l2 = self.g2(x2)                  # (B,1,H2,W2)
        l3 = self.g3(x3_up)                # (B,1,H2,W2)
        wa = th.softmax(th.cat([l2, l3], dim=1) / self.T, dim=1)  # (B,2,H2,W2)
        w2, w3 = wa[:, :1], wa[:, 1:]

        f1 = th.cat([
            self.alpha2 * w2.expand_as(x2)   * x2,
            self.alpha3 * w3.expand_as(x3_up) * x3_up
        ], dim=1)     # (B, C2+C3, H2, W2)

        # ---------- Stage B: x1 ⟷ (f1→H/2) ----------
        f1_up = F.interpolate(f1, (H1, W1), mode='bilinear', align_corners=False)
        l1   = self.g1(x1)                # (B,1,H1,W1)
        lf1  = self.gf1(f1_up)             # (B,1,H1,W1)
        wb = th.softmax(th.cat([l1, lf1], dim=1) / self.T, dim=1)  # (B,2,H1,W1)
        w1, wf1 = wb[:, :1], wb[:, 1:]

        f2 = th.cat([
            self.alpha1  * w1.expand_as(x1)   * x1,
            self.alphaf1 * wf1.expand_as(f1_up)* f1_up
        ], dim=1)  # (B, C1 + (C2+C3), H1, W1)

        # ---------- Stage C: x0 ⟷ (f2→H) ----------
        f2_up = F.interpolate(f2, (H, W), mode='bilinear', align_corners=False)
        l0   = self.g0(x0)
        lf2  = self.gf2(f2_up)
        wc = th.softmax(th.cat([l0, lf2], dim=1) / self.T, dim=1)  # (B,2,H,W)
        w0, wf2 = wc[:, :1], wc[:, 1:]

        x = th.cat([
            self.alpha0  * w0.expand_as(x0)   * x0,
            self.alphaf2 * wf2.expand_as(f2_up)* f2_up
        ], dim=1)  # (B, C0 + C1 + C2 + C3, H, W)

        return self.out_layer(x)



class MultiScaleConcatWeightedV2(nn.Module):
    def __init__(self, in_chs=[256, 512, 1024], out_dim=256, mid=256, T=1.0):
        """
        Learned weighted fusion for 3 scales (x0, x1, x2).
        Stage A: x1 (H/2) ⟷ up(x2)->(H/2)
        Stage B: x0 (H)   ⟷ up(f1)->(H)
        """
        super().__init__()
        C0, C1, C2 = in_chs
        self.T = T

        # --- Gating heads (branch logits: 1ch) ---
        # Stage A: x1 vs up(x2)
        self.g1   = nn.Conv2d(C1, 1, kernel_size=1)          # logits for x1
        self.g2u  = nn.Conv2d(C2, 1, kernel_size=1)          # logits for up(x2)

        # Stage B: x0 vs up(f1)  (f1 has C1+C2 channels)
        self.g0   = nn.Conv2d(C0, 1, kernel_size=1)          # logits for x0
        self.gf1u = nn.Conv2d(C1 + C2, 1, kernel_size=1)     # logits for up(f1)

        # --- Global scalar gates (learnable per-branch) ---
        self.alpha2  = nn.Parameter(th.tensor(1.0))   # x2 branch (after upsample)
        self.alpha1  = nn.Parameter(th.tensor(1.0))   # x1 branch
        self.alphaf1 = nn.Parameter(th.tensor(1.0))   # f1 branch (after upsample)
        self.alpha0  = nn.Parameter(th.tensor(2.0))   # x0 branch

        # --- Output head ---
        # Input channels = C0 + C1 + C2  (x0 concat up(f2) where f2 is x1+up(x2))
        self.out_layer = nn.Sequential(
            nn.Conv2d(C0 + C1 + C2, 2*mid, kernel_size=1, bias=False),
            nn.GroupNorm(32, 2*mid),
            nn.SiLU(inplace=True),

            nn.Conv2d(2*mid, 2*mid, kernel_size=3, padding=1, groups=2*mid, bias=False),
            nn.GroupNorm(32, 2*mid),
            nn.SiLU(inplace=True),

            nn.Conv2d(2*mid, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, out_dim),
        )

        # Balanced init
        nn.init.zeros_(self.g1.bias);   nn.init.zeros_(self.g2u.bias)
        nn.init.zeros_(self.g0.bias);   nn.init.zeros_(self.gf1u.bias)

    def forward(self, xs):
        # xs: (x0, x1, x2)
        x0, x1, x2 = xs
        B, _, H,  W  = x0.shape
        _, _, H1, W1 = x1.shape

        # ---------- Stage A: x1 ⟷ up(x2) at (H1, W1) ----------
        x2u = F.interpolate(x2, (H1, W1), mode='bilinear', align_corners=False)
        l1  = self.g1(x1)                      # (B,1,H1,W1)
        l2u = self.g2u(x2u)                    # (B,1,H1,W1)
        wa  = th.softmax(th.cat([l1, l2u], dim=1) / self.T, dim=1)   # (B,2,H1,W1)
        w1, w2 = wa[:, :1], wa[:, 1:]

        # weighted-concat at stage A
        f1 = th.cat([
            self.alpha1 * w1.expand_as(x1)  * x1,
            self.alpha2 * w2.expand_as(x2u) * x2u
        ], dim=1)   # (B, C1+C2, H1, W1)

        # ---------- Stage B: x0 ⟷ up(f1) at (H, W) ----------
        f1u = F.interpolate(f1, (H, W), mode='bilinear', align_corners=False)
        l0   = self.g0(x0)                     # (B,1,H,W)
        lf1u = self.gf1u(f1u)                  # (B,1,H,W)
        wb   = th.softmax(th.cat([l0, lf1u], dim=1) / self.T, dim=1) # (B,2,H,W)
        w0, wf1 = wb[:, :1], wb[:, 1:]

        x = th.cat([
            self.alpha0  * w0.expand_as(x0)  * x0,
            self.alphaf1 * wf1.expand_as(f1u)* f1u
        ], dim=1)   # (B, C0 + C1 + C2, H, W)

        return self.out_layer(x)
    
    
class MultiScaleConcatThree(nn.Module):
    def __init__(self, in_chs=[256,512,1024], out_dim=256, mid=256):
        super().__init__()

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(in_chs), 2*mid, kernel_size=1, bias=False),
            nn.GroupNorm(32, 2*mid),           
            nn.SiLU(inplace=True),

            nn.Conv2d(2*mid, 2*mid, kernel_size=3, padding=1, groups=2*mid, bias=False),
            nn.GroupNorm(32, 2*mid),
            nn.SiLU(inplace=True),

            nn.Conv2d(2*mid, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, out_dim),       
        )

    def forward(self, xs):
        x0, x1, x2 = xs
        B, _, H, W = x0.shape
        B, _, H1, W1 = x1.shape
                    
        f1 = th.cat([x1, F.interpolate(x2, (H1, W1), mode='bilinear', align_corners=False)], dim=1)     # H//2, W//2
        f2 = th.cat([x0, F.interpolate(f1, (H, W), mode='bilinear', align_corners=False)], dim=1)     # H, W  
        return self.out_layer(f2)           # [B,out_dim,H,W]
