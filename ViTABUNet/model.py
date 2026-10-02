import torch
import torch.nn as nn
import torch.nn.functional as F
from .mix_transformer import MixVisionTransformer
from functools import partial
import warnings
import os

class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        self.positional_encodings = nn.Parameter(torch.rand(1500, embedding_dim), requires_grad=True)

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2)
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)
        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))
        return y.permute(0, 2, 1).view(n, cout, h, w)

class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())
        x = self.conv3x3(x)
        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)

        y = self.regressor(regression_head)
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps
        
class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features, input_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())
        self.conv_transpose = nn.ConvTranspose2d(input_features, input_features, 2, stride=2)

    def forward(self, x, concat_with):
        up_x = self.conv_transpose(x)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)

class DecoderBN(nn.Module):
    def __init__(self, in_channels=[64, 128, 320, 512], num_classes = 256):
        super(DecoderBN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels[3], in_channels[3], kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_channels[3]), nn.LeakyReLU())

        self.up1 = UpSampleBN(skip_input=in_channels[3]+in_channels[2], output_features=in_channels[2], input_features=in_channels[3])
        self.up2 = UpSampleBN(skip_input=in_channels[2]+in_channels[1], output_features=in_channels[1], input_features=in_channels[2])
        self.up3 = UpSampleBN(skip_input=in_channels[1]+in_channels[0], output_features=in_channels[0], input_features=in_channels[1])
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels[0], in_channels[0],  2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels[0], in_channels[1], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(in_channels[1]),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(in_channels[1], in_channels[1], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(in_channels[1]),
                                  nn.LeakyReLU())
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels[0], in_channels[0], 2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(in_channels[0]),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(in_channels[0]),
                                  nn.LeakyReLU(),
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels[0], num_classes, kernel_size=1, stride=1, padding=0))
        self.activate =  nn.Softmax(dim=1)

        self.regressor = nn.Sequential(nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256))

    def forward(self, features):
        x_block1, x_block2, x_block3, x_block4 = features
        x_d0 = self.conv1(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)

        x_d3 = self.conv_transpose1(x_d3)
        x_d4 = self.conv2(x_d3)
        return x_d4

class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.model1 = backend

    def forward(self, x):
        out_model1 = self.model1(x)
        return out_model1

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class UnetAdaptiveBins(nn.Module):
    def __init__(self, backend, n_bins=100, min_val=0.1, max_val=10, norm='linear'):
        super(UnetAdaptiveBins, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = Encoder(backend)
        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=128, norm=norm)
        self.decoder = DecoderBN(in_channels=[64, 128, 320, 512])
        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

    def forward(self, x, **kwargs):
        enc  = self.encoder(x)

        unet_out = self.decoder(enc, **kwargs)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(unet_out)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        return centers.view(n, dout), pred, out


    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, encoder_weights="pre-trained-weights/mit_b4.pth", **kwargs):
        model1 = mit_b4(**kwargs)
        if encoder_weights is not None:
            state_dict = torch.load(encoder_weights, map_location='cpu', weights_only=True)
            state_dict.pop('head.weight', None)
            state_dict.pop('head.bias', None)
            model1.load_state_dict(state_dict, strict=False)

        # Building Encoder-Decoder model
        m = cls(model1, n_bins=n_bins, **kwargs)
        return m
