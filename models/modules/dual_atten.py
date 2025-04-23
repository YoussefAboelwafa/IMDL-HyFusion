from torch import nn
import torch



class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DAHead, self).__init__()
        self.aux = aux
        
        # Create separate attention modules for each feature level
        self.attention_modules = nn.ModuleList()
        
        # Handle input as list of channel sizes if it's a list
        if isinstance(in_channels, list):
            self.feature_levels = len(in_channels)
            for channels in in_channels:
                attention_module = self._create_attention_module(
                    channels, channels, channels, norm_layer, norm_kwargs, **kwargs
                )
                self.attention_modules.append(attention_module)
        else:
            self.feature_levels = 1
            attention_module = self._create_attention_module(
                in_channels, in_channels, in_channels, norm_layer, norm_kwargs, **kwargs
            )
            self.attention_modules.append(attention_module)

    def _create_attention_module(self, in_channels, inter_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        return nn.ModuleDict({
            'conv_p1': nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(True)
            ),
            'conv_c1': nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(True)
            ),
            'pam': _PositionAttentionModule(inter_channels, **kwargs),
            'cam': _ChannelAttentionModule(**kwargs),
            'conv_p2': nn.Sequential(
                nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(True)
            ),
            'conv_c2': nn.Sequential(
                nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(True)
            )
        })

    def _process_single_level(self, x, module):
        feat_p = module['conv_p1'](x)
        feat_p = module['pam'](feat_p)
        feat_p = module['conv_p2'](feat_p)

        feat_c = module['conv_c1'](x)
        feat_c = module['cam'](feat_c)
        feat_c = module['conv_c2'](feat_c)

        feat_fusion = feat_p + feat_c + x  # Add residual connection
        
        return feat_fusion

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
            
        outputs = []
        for idx, feature in enumerate(x):
            enhanced_feature = self._process_single_level(feature, self.attention_modules[idx])
            outputs.append(enhanced_feature)
            
        return outputs if len(outputs) > 1 else outputs[0]