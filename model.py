python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexSpatialAttentionModule(nn.Module):
    def __init__(self, channel):
        super(ComplexSpatialAttentionModule, self).__init__()
        self.query_conv = nn.Conv2d(channel, channel // 8, 1)
        self.key_conv = nn.Conv2d(channel, channel // 8, 1)
        self.value_conv = nn.Conv2d(channel, channel, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(channel, channel, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        query = query.view(B, -1, H * W).permute(0, 2, 1)
        key = key.view(B, -1, H * W)
        attention = self.softmax(torch.bmm(query, key))
        value = value.view(B, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.out_conv(out) + x
        return out
