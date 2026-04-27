import torch
import torch.nn as nn

from tok.ar_dtok.vqvae import VQVAE
from tok.rar_dtok.rar_model import RARCondModel
from tok.ta_tok import TextAlignedTokenizer


class RARMMAutoEncoder(nn.Module):
    def __init__(
        self,
        rar_path,
        encoder_path,
        decoder_path,
        encoder_args={},
        decoder_args={},
    ):
        super().__init__()
        self.rar_model = RARCondModel.from_checkpoint(rar_path)
        self.encoder = TextAlignedTokenizer.from_checkpoint(encoder_path, load_teacher=False, **encoder_args)
        self.decoder = VQVAE.from_checkpoint(decoder_path, **decoder_args)

    def rar_sample(self, x, args):
        return self.rar_model.sample(
            x,
            cfg_scale=args.get("cfg_scale", 1.0),
            cfg_interval=args.get("cfg_interval", -1),
            temperature=args.get("temperature", 1.0),
            top_k=args.get("top_k", 0),
            top_p=args.get("top_p", 1.0),
            guidance_scale_pow=args.get("guidance_scale_pow", 2.5),
        )

    def post_process(self, x):
        x = x.cpu().float().clamp(0.0, 1.0) * 255.0
        x = x.permute(0, 2, 3, 1)
        return x.to(torch.uint8)

    def encode(self, x):
        return self.encoder(x.to(self.encoder.dtype))["encoded"]

    def get_encoder_indices(self, x):
        return self.encoder(x.to(self.encoder.dtype))["bottleneck_rep"]

    @torch.inference_mode()
    def decode_from_encoder_indices(self, indices, args={}):
        encoder_x = self.encoder.decode_from_bottleneck(indices)
        encoder_x = encoder_x.to(device=self.rar_model.device, dtype=self.rar_model.dtype)
        rar_indices = self.rar_sample(encoder_x, args)
        decoder_x = self.decoder.decode_from_bottleneck(rar_indices)
        return self.post_process(decoder_x)

    def decode_from_vqvae_indices(self, indices):
        decoder_x = self.decoder.decode_from_bottleneck(indices)
        return self.post_process(decoder_x)

    @torch.inference_mode()
    def forward(self, x, args={}):
        encoder_x = self.encoder(x.to(self.encoder.dtype))["encoded"]
        encoder_x = encoder_x.to(device=self.rar_model.device, dtype=self.rar_model.dtype)
        rar_indices = self.rar_sample(encoder_x, args)
        decoder_x = self.decoder.decode_from_bottleneck(rar_indices)
        return self.post_process(decoder_x)
