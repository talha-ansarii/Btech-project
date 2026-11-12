"""Baseline networks.

RGBDFusionNet (mode="rgbd")
    ESANet-style design (Seichter et al., "Efficient RGB-D Semantic Segmentation for
    Indoor Scene Analysis", ICRA 2021): two ImageNet-pretrained ResNet-34 encoders,
    one for RGB and one for depth. After every stage the depth features are added
    into the RGB stream, each side first re-weighted by squeeze-and-excitation
    (channel attention). A light FPN-style decoder with a pooling context module
    predicts the 40 classes at 1/4 resolution, then upsamples to full size.

RGBDFusionNet (mode="rgb")
    The same network with the depth branch removed. Needed later as the
    "RGB-only proxy" when we score how much depth adds for each image.

Both expose `forward(rgb, depth, return_feats=False)`.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(c // r, 8), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(c // r, 8), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class SEFusion(nn.Module):
    """fused = SE_rgb(rgb) + SE_depth(depth)   (ESANet fusion block)"""

    def __init__(self, c):
        super().__init__()
        self.se_rgb, self.se_depth = SqueezeExcite(c), SqueezeExcite(c)

    def forward(self, rgb, depth):
        return self.se_rgb(rgb) + self.se_depth(depth)


def conv_bn_relu(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class PoolingContext(nn.Module):
    """Pyramid pooling (PSPNet-style) on the deepest features, for global scene context."""

    def __init__(self, cin, cout, bins=(1, 2, 4)):
        super().__init__()
        self.stages = nn.ModuleList(
            [nn.Sequential(nn.AdaptiveAvgPool2d(b), conv_bn_relu(cin, cin // 4, 1)) for b in bins])
        self.out = conv_bn_relu(cin + len(bins) * (cin // 4), cout, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [x] + [F.interpolate(s(x), (h, w), mode="bilinear", align_corners=False) for s in self.stages]
        return self.out(torch.cat(feats, 1))


class RGBDFusionNet(nn.Module):
    def __init__(self, num_classes=40, mode="rgbd", backbone="resnet34", pretrained=True, dec_ch=128):
        super().__init__()
        assert mode in ("rgbd", "rgb")
        self.mode = mode
        kw = dict(features_only=True, out_indices=(0, 1, 2, 3, 4), pretrained=pretrained)
        self.enc_rgb = timm.create_model(backbone, **kw)
        chs = self.enc_rgb.feature_info.channels()  # resnet34: [64, 64, 128, 256, 512]
        if mode == "rgbd":
            # in_chans=1: timm sums the pretrained RGB filters into one depth channel
            self.enc_depth = timm.create_model(backbone, in_chans=1, **kw)
            self.fusions = nn.ModuleList([SEFusion(c) for c in chs])
        self.context = PoolingContext(chs[4], dec_ch)
        self.lateral = nn.ModuleList([conv_bn_relu(c, dec_ch, 1) for c in chs[1:4]])
        self.smooth = nn.ModuleList([conv_bn_relu(dec_ch, dec_ch) for _ in chs[1:4]])
        self.head = nn.Sequential(conv_bn_relu(dec_ch, dec_ch), nn.Dropout2d(0.1),
                                  nn.Conv2d(dec_ch, num_classes, 1))

    @staticmethod
    def _stages(enc):
        """Yield the 5 stage outputs of a timm ResNet one by one (so we can fuse in between)."""
        return [nn.Sequential(enc.conv1, enc.bn1, enc.act1),
                nn.Sequential(enc.maxpool, enc.layer1), enc.layer2, enc.layer3, enc.layer4]

    def encode(self, rgb, depth=None):
        feats = []
        if self.mode == "rgb":
            return self.enc_rgb(rgb)
        x, d = rgb, depth
        for s_rgb, s_d, fuse in zip(self._stages(self.enc_rgb), self._stages(self.enc_depth), self.fusions):
            x, d = s_rgb(x), s_d(d)
            x = fuse(x, d)          # depth flows into the RGB stream; depth stream continues as is
            feats.append(x)
        return feats

    def forward(self, rgb, depth=None, return_feats=False):
        size = rgb.shape[-2:]
        f = self.encode(rgb, depth)                       # strides 2, 4, 8, 16, 32
        y = self.context(f[4])
        for i in (2, 1, 0):                               # top-down: 1/16 -> 1/8 -> 1/4
            lat = self.lateral[i](f[i + 1])
            y = F.interpolate(y, lat.shape[-2:], mode="bilinear", align_corners=False) + lat
            y = self.smooth[i](y)
        logits = F.interpolate(self.head(y), size, mode="bilinear", align_corners=False)
        return (logits, y) if return_feats else logits


if __name__ == "__main__":
    for mode in ("rgbd", "rgb"):
        m = RGBDFusionNet(mode=mode, pretrained=False)
        out = m(torch.randn(2, 3, 480, 640), torch.randn(2, 1, 480, 640))
        print(mode, tuple(out.shape), f"{sum(p.numel() for p in m.parameters()) / 1e6:.1f}M params")
