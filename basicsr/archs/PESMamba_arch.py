from typing import Tuple, List
from torch import Tensor
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY


#############################
# ResGroup with SDFF Fusion
#############################
class SDFFResGroup(nn.Module):
    """ResGroup variant that uses SDFF for fusing global (SME) and local (RME) features.

    SME and RME run in parallel, then their outputs are fused via SDFF
    (Spatial-Dynamic Feature Fusion: Channel Attention + Spatial Attention + Pixel Attention gating),
    followed by a 1x1 projection to the original channel dimension.
    """
    def __init__(self,
                 in_ch: int,
                 attn_drop_rate: float = 0,
                 d_state: int = 16,
                 K: int = 4):
        super().__init__()

        self.global_block = SME(in_ch=in_ch,
                                d_state=d_state,
                                attn_drop_rate=attn_drop_rate)

        self.local_block = RME(in_ch=in_ch, K=K)

        # SDFF: fuses two feature maps x, y → outputs 2*in_ch
        self.fusion = SDFF(dim=in_ch)

        # Projection merged into SDFF.conv (now outputs dim directly)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_global = self.global_block(x)
        x_local = self.local_block(x)
        # SDFF expects (x, y) as two separate feature maps
        x = self.fusion(x_global, x_local)
        return x


######################
# Meta Architecture
######################
@ARCH_REGISTRY.register()
class PESMamba(nn.Module):
    def __init__(self,
                 scale: int = 4,
                 in_chans: int = 3,
                 num_layers: int = 4,
                 embedding_dim: int = 64,
                 img_range: float = 1.0,
                 attn_drop_rate: float = 0,
                 d_state: int = 16,
                 K_list: List[int] = None,
                 ):
        super().__init__()
        self.scale = scale
        self.num_in_channels = in_chans
        self.num_out_channels = in_chans
        self.img_range = img_range
        self.num_layers = num_layers

        if K_list is None:
            K_list = [4] * num_layers

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        # -- SHALLOW FEATURES --
        self.conv_1 = nn.Conv2d(self.num_in_channels, embedding_dim, kernel_size=3, padding=1)

        # -- DEEP FEATURES --
        self.body = nn.ModuleList(
            [SDFFResGroup(in_ch=embedding_dim,
                          attn_drop_rate=attn_drop_rate,
                          d_state=d_state,
                          K=K_list[i]
                          ) for i in range(num_layers)]
        )

        # -- HIERARCHICAL ADAPTIVE RESIDUAL --
        self.hier_weights = nn.Parameter(torch.zeros(num_layers))
        self.hier_scale = nn.Parameter(torch.tensor(0.1))

        # -- UPSCALE --
        self.norm = LayerNorm(embedding_dim, data_format='channels_first')
        self.conv_2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(embedding_dim, (scale**2) * self.num_out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # -- SHALLOW FEATURES --
        x = self.conv_1(x)
        res = x

        # -- DEEP FEATURES --
        hier_features = []
        for idx, layer in enumerate(self.body):
            x = layer(x)
            hier_features.append(x)

        # Hierarchical adaptive residual
        w = F.softmax(self.hier_weights, dim=0)
        hier_out = sum(w[i] * hier_features[i] for i in range(self.num_layers))
        x = x + self.hier_scale * hier_out

        x = self.norm(x)

        # -- HR IMAGE RECONSTRUCTION --
        x = self.conv_2(x) + res
        x = self.upsampler(x)

        x = x / self.img_range + self.mean
        return x


#############################
# Components
#############################    
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=False),
            nn.BatchNorm2d(dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)
        pattn1 = pattn1.unsqueeze(dim=2)
        x2 = torch.cat([x, pattn1], dim=2)
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class SDFF(nn.Module):
    def __init__(self, dim, reduction=8):
        super(SDFF, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial_add = x + y
        cattn = self.ca(initial_add)
        sattn = self.sa(initial_add)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial_add, pattn1))

        # Remove skip
        result0 = pattn2 * x
        result1 = (1 - pattn2) * y

        # Cat+Conv
        result = torch.cat([result0, result1], dim=1)
        result = self.conv(result)

        return result


class ResGroup(nn.Module):
    def __init__(self,
                 in_ch: int,
                 attn_drop_rate: float = 0,
                 d_state: int = 16,
                 K: int = 4):
        super().__init__()

        self.global_block = SME(in_ch=in_ch,
                                d_state=d_state,
                                attn_drop_rate=attn_drop_rate)

        self.local_block = RME(in_ch=in_ch, K=K)

        self.fusion = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_global = self.global_block(x)
        x_local = self.local_block(x)
        x = self.fusion(torch.cat([x_global, x_local], dim=1))
        return x


class StridedReflectConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=0,
                              groups=ch, bias=False)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        return self.conv(x)


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.pool = StridedReflectConv(self.d_inner)
        self.res_scale = nn.Parameter(torch.ones(1, self.d_inner, 1, 1))
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, C, H_orig, W_orig = x.shape

        # 1. original input projection
        x_flat = x.flatten(2).transpose(1, 2)  # (B, L, d_model)
        xz = self.in_proj(x_flat)
        x_inner, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)

        # 2. local conv branch
        x_conv = x_inner.transpose(1, 2).contiguous().view(B, -1, H_orig, W_orig)
        x_conv = self.act(self.conv2d(x_conv))

        # 3. Pad to even for lossless 2x down/up — preserves ALL original pixels
        pad_h = (2 - H_orig % 2) % 2
        pad_w = (2 - W_orig % 2) % 2
        x_pad = F.pad(x_conv, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h + pad_w > 0) else x_conv
        H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]  # always even

        xlow = self.pool(x_pad)
        # StridedReflectConv: pad(1,1) → 3×3 s=2 → exact H_pad//2 × W_pad//2

        # Residual in even space (lossless: H_pad,W_pad are 2× of xlow)
        res_pad = x_pad - F.interpolate(xlow, (H_pad, W_pad), mode='nearest')

        # 4. core scan on downsampled features
        y1, y2, y3, y4 = self.forward_core(xlow)

        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous()

        # 5. lossless 2x upsample (H_pad//2 → H_pad, exact integer scaling)
        y = y.transpose(1, 2).contiguous().view(B, self.d_inner, H_pad // 2, W_pad // 2)
        y = F.interpolate(y, scale_factor=2, mode='bilinear') + self.res_scale * res_pad

        # 6. Crop back to original size (discard padded border)
        y = y[:, :, :H_orig, :W_orig]
        y = y.flatten(2).transpose(1, 2)

        # 7. output norm + gate + projection
        y = self.out_norm(y)
        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(B, -1, H_orig, W_orig)
        return out


class SME(nn.Module):
    def __init__(self,
                 in_ch: int,
                 attn_drop_rate: float = 0,
                 d_state: int = 16,
                 expand: float = 2.,
                 **kwargs):
        super().__init__()

        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = SS2D(d_model=in_ch, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)

        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm_1(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x


#############################
# Local Blocks
#############################
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepDW(torch.nn.Module):
    """RepVGG-style Depthwise 模块.
    
    三路并行:
      - conv:  3×3 DW + BN
      - conv1: 1×1 per-channel conv
      - identity shortcut
    三路相加后经过输出 BN.
    推理时可融合为单个 3×3 DW Conv.
    """
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
        self.apply(self._init_weights)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class SPCluster(nn.Module):

    def __init__(self, dim, K=4):
        super().__init__()
        self.dim = dim
        self.K = K

        # ★ V13: K 个独立前置 PW (1×1 Conv) —— 每组独立通道映射
        self.pw_pre = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
            for _ in range(K)
        ])

        # ★ V13: 1 个共享后置 PW 
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        # ★ K 个独立 3×3 conv, 每个输出 1 通道 spatial attention map
        self.conv_attn = nn.ModuleList([
            nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=False)
            for _ in  range(K)
        ])

        # ★ K 组独立 PLE 空洞密集残差
        #   每组: RepDW (d=1) → DW3×3 d=2 → DW3×3 d=3
        self.dw_groups = nn.ModuleList([
            nn.ModuleList([
                # dw1: RepDW (三路并行: 3×3 DW + 1×1 per-channel + identity)
                RepDW(dim),
                # dw2: dilated DW d=2, 感受野 5×5
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2,
                          groups=dim, bias=False),
                # dw3: dilated DW d=3, 感受野 7×7
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3, dilation=3,
                          groups=dim, bias=False),
            ]) for _ in range(K)
        ])
        self.global_branch = nn.Sequential(
                    RepDW(dim),
                    nn.GELU(),
                    RepDW(dim),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )
         # 激活函数
        self.gelu = nn.GELU()

        # 温度系数 (softmax 锐度)
        self.log_tau = nn.Parameter(torch.zeros(1))

        # 输出归一化
        self.norm = nn.LayerNorm(dim)

    def _compute_masks(self, x):
        """
        K 个 conv 生成 K 个空间 attention map → 加权池化 → centers
        再计算像素到各中心的软分配 mask

        x: (B, C, H, W)
        返回: (B, K, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        x_flat = x.view(B, C, N).transpose(1, 2)            # (B, N, C)
        x_flat_norm = F.normalize(x_flat, p=2, dim=-1)      # (B, N, C)

        # ① K 个 conv → K 个 spatial attention map → 加权池化得到 K 个 centers
        centers_list = []
        for conv_k in self.conv_attn:
            attn_k = conv_k(x)                               # (B, 1, H, W)
            attn_k = attn_k.view(B, 1, N)                    # (B, 1, N)
            attn_k = F.softmax(attn_k, dim=-1)               # softmax over spatial positions
            center_k = torch.matmul(attn_k, x_flat)          # (B, 1, C) — weighted sum pooling
            centers_list.append(center_k)

        centers = torch.cat(centers_list, dim=1)             # (B, K, C)

        # ② Q = softmax(cos_sim(x, centers) / tau)
        c_norm = F.normalize(centers, p=2, dim=-1)           # (B, K, C)
        similarity = torch.matmul(x_flat_norm, c_norm.transpose(1, 2))  # (B, N, K)

        tau = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
        masks = F.softmax(similarity / tau, dim=-1)           # (B, N, K)

        masks = masks.view(B, H, W, self.K).permute(0, 3, 1, 2)   # (B, K, H, W)
        return masks

    def forward(self, x):
        identity = x  # ★ PLE 大残差连接
        global_feat = self.global_branch(x)  
        # ★ 计算语义掩码 (直接在输入特征上计算，无需 conv_pre)
        masks = self._compute_masks(x)  # (B, K, H, W)

        # ★ 每组独立处理 (V13: PW 前置 + PLE 空洞密集残差)
        out = 0
        for k in range(self.K):
            mask_k = masks[:, k:k+1]                    # (B, 1, H, W)
            cur_x = x * mask_k                           # ★ 掩码门控 (直接用 x，无需 x_hat)

            # ★ V13: 每组独立前置 PW → 语义自适应通道映射
            X_k = self.pw_pre[k](cur_x)

            # ★ PLE-style 密集空洞残差 (纯 DW，无层间 PW/GELU 打断)
            dw1, dw2, dw3 = self.dw_groups[k]
            X1 = dw1(X_k)                                   # RepDW: 三路并行, d=1
            X2 = dw2(X_k + X1)                              # DW d=2, 密集残差
            X3 = dw3(X_k + X1 + X2) 
            out = out + X_k + X1 + X2 + X3

            
        cluster_feat = self.gelu(out)
        concat_feat = torch.cat([global_feat, cluster_feat], dim=1)
        out = self.fusion(concat_feat)

        out = out + identity

        # LayerNorm: (B, C, H, W) → permute → LN → permute back
        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()



        return out


class RME(nn.Module):
    def __init__(self,
                 in_ch: int,
                 K: int = 4,):
        super().__init__()

        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = SPCluster(dim=in_ch, K=K)

        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm_1(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x


#################
# MoE Layer
#################
class MoEBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 use_shuffle: bool = False,
                 num_heads: int = 3,
                 recursive: int = 2):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.recursive = recursive
        self.num_heads = num_heads
        self.head_dim = in_ch // num_heads

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 2*in_ch, kernel_size=1, padding=0)
        )

        self.agg_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, groups=in_ch),
            nn.GELU())

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            # StripedConv2d(in_ch, kernel_size=5, depthwise=True),
            nn.GELU())

        self.k_proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, groups=num_heads),
            nn.Sigmoid()
        )

        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)

    def calibrate(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x

        for _ in range(self.recursive):
            x = self.agg_conv(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return self.k_proj(res + x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)

        if self.use_shuffle:
            x = channel_shuffle(x, groups=2)
        x, k = torch.chunk(x, chunks=2, dim=1)

        x = self.conv_2(x)
        k = self.calibrate(k)

        x = x * k
        x = self.proj(x)
        return x


#################
# Utilities
#################
class StripedConv2d(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int,
                 depthwise: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(1, self.kernel_size), padding=(0, self.padding), groups=in_ch if depthwise else 1),
            nn.Conv2d(in_ch, in_ch, kernel_size=(self.kernel_size, 1), padding=(self.padding, 0), groups=in_ch if depthwise else 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x


class GatedFFN(nn.Module):
    def __init__(self,
                 in_ch,
                 mlp_ratio,
                 kernel_size,
                 act_layer,):
        super().__init__()
        mlp_ch = in_ch * mlp_ratio

        self.fn_1 = nn.Sequential(
            nn.Conv2d(in_ch, mlp_ch, kernel_size=1, padding=0),
            act_layer,
        )
        self.fn_2 = nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)

        self.gate = nn.Conv2d(mlp_ch // 2, mlp_ch // 2,
                              kernel_size=kernel_size, padding=kernel_size // 2, groups=mlp_ch // 2)

    def feat_decompose(self, x):
        s = x - self.gate(x)
        x = x + self.sigma * s
        return x

    def forward(self, x: torch.Tensor):
        x = self.fn_1(x)
        x, gate = torch.chunk(x, 2, dim=1)

        gate = self.gate(gate)
        x = x * gate

        x = self.fn_2(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


if __name__ == '__main__':
    model = SDFFMambaV10(scale=4).cuda()
    x = torch.randn(2, 3, 128, 128).cuda()
    y = model(x)
    print(f'Input:  {tuple(x.shape)}')
    print(f'Output: {tuple(y.shape)}')
    print(model.__repr__())