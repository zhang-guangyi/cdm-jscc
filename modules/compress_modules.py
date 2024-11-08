import torch.nn as nn
from .network_components import ResnetBlock, FlexiblePrior, Downsample, Upsample
from .utils import quantize, NormalDistribution
import torch
from layer.jscc_encoder import JSCCEncoder
from layer.jscc_decoder import JSCCDecoder
from channel.channel import Channel
from datetime import datetime

class Compressor(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1), 
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        snr=10,
        mask_ratio=0.13,
    ):
        super().__init__()
        self.snr = snr
        self.mask_ratio=mask_ratio
        self.channels = channels
        self.out_channels = out_channels
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))
        self.reversed_dims = [*map(lambda m: dim * m, reverse_dim_mults), out_channels]
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))
        assert self.dims[-1] == self.reversed_dims[0]
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)]
        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:]))
        self.reversed_hyper_dims = list(
            reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)])
        )
        self.reversed_hyper_in_out = list(
            zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:])
        )
        self.prior = FlexiblePrior(self.hyper_dims[-1])
        self.fe = JSCCEncoder()
        self.fd = JSCCDecoder()
        self.channel = Channel(type='awgn', param = snr)
        self.mse = nn.MSELoss()

    def get_extra_loss(self):
        return self.prior.get_extraloss()

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

    def encode(self, input):
        for i, (resnet, down) in enumerate(self.enc):
            input = resnet(input)
            input = down(input)
        latent = input
        for i, (conv, act) in enumerate(self.hyper_enc):
            input = conv(input)
            input = act(input)
        hyper_latent = input
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        input = q_hyper_latent
        for i, (deconv, act) in enumerate(self.hyper_dec):
            input = deconv(input)
            input = act(input)

        mean, scale = input.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent, 
            "latent_distribution": latent_distribution,
        }
        return q_latent, q_hyper_latent, state4bpp

    def decode(self, input):
        output = []
        for i, (resnet, up) in enumerate(self.dec):
            input = resnet(input)
            input = up(input)
            output.append(input)
        return output[::-1]

    def bpp(self, shape, state4bpp):
        B, _, H, W = shape
        latent = state4bpp["latent"]
        hyper_latent = state4bpp["hyper_latent"]
        latent_distribution = state4bpp["latent_distribution"]
        if self.training:
            q_hyper_latent = quantize(hyper_latent, "noise")
            q_latent = quantize(latent, "noise")
        else:
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        
        latent_likelihoods = latent_distribution.likelihood(q_latent)
        cond_rate = -latent_likelihoods.log2()
        hyper_likelihoods = self.prior.likelihood(q_hyper_latent)
        hyper_rate = -hyper_likelihoods.log2()

        bpp = (hyper_rate.sum(dim=(1, 2, 3)) + cond_rate.sum(dim=(1, 2, 3))) / (H * W)
        return bpp, latent, latent_likelihoods

    def update_resolution(self, H, W):
        # Update attention mask for W-MSA and SW-MSA
        self.fe.update_resolution(H // 16, W // 16)
        self.fd.update_resolution(H // 16, W // 16)

    def forward(self, input):
        # print(self.snr)
        B, C, H, W = input.shape
        num_pixels = H * W * C
        self.update_resolution(H, W)  


        q_latent, q_hyper_latent, state4bpp = self.encode(input)
        bpp, latent, latent_likelihoods = self.bpp(input.shape, state4bpp) 

        # ********************************JSCC forward******************************* # 
        s_masked, mask_BCHW, indexes = self.fe(latent, latent_likelihoods.detach(), eta=self.mask_ratio) 
        output = self.decode(q_latent)

        # Pass through the channel.
        mask_BCHW = mask_BCHW.byte()
        mask_BCHW = mask_BCHW.bool()
        channel_input = torch.masked_select(s_masked, mask_BCHW)
        # print(datetime.now())
        channel_output, channel_usage = self.channel.forward(channel_input)
        s_hat = torch.zeros_like(s_masked)
        s_hat[mask_BCHW] = channel_output
        cbr_y = channel_usage / num_pixels
        latent_hat = self.fd(s_hat, indexes) 
        output_hat = self.decode(latent_hat)

        
        return {
            "output": output,
            "output_hat": output_hat,
            "bpp": bpp,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
            "cbr_y":cbr_y,
        }


class ResnetCompressor(Compressor):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        snr = 10,
        mask_ratio = 0.13,
    ):
        super().__init__(
            dim,
            dim_mults, 
            reverse_dim_mults,
            hyper_dims_mults,
            channels,
            out_channels,
            snr,
            mask_ratio
        )
        self.build_network()

    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        Downsample(dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                        Upsample(dim_out if not is_last else dim_in, dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )
