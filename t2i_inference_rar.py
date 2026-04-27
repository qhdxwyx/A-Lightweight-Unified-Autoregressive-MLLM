import re
import argparse
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

import llava.model  # noqa: F401
from llava.utils import apply_chat_template
from tok.rar_autoencoder import RARMMAutoEncoder


@dataclass
class T2IRARConfig:
    model_path: str = "output_dir/tar_qwen3_0.6B_pretrain_last"
    rar_path: str = "output_dir/latest.pth"
    encoder_path: str = "assets/common/ta_tok.pth"
    decoder_path: str = "assets/common/vq_ds16_t2i.pt"
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    scale: int = 0
    seq_len: int = 729
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 1200
    cfg_scale: float = 4.0
    guidance_scale_pow: float = 2.5
    max_new_tokens: int = 729


class TextToImageRARInference:
    def __init__(self, config: T2IRARConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.set_grad_enabled(False)
        self._load_models()

    def _load_models(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=self.config.dtype,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, fix_mistral_regex=True)

        self.visual_tokenizer = RARMMAutoEncoder(
            rar_path=self.config.rar_path,
            encoder_path=self.config.encoder_path,
            decoder_path=self.config.decoder_path,
            encoder_args={"input_type": "rec"},
            decoder_args={},
        ).eval().to(dtype=self.config.dtype, device=self.device)

    def _prompt_to_input_ids(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        input_text = apply_chat_template(
            self.tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_text += f"<im_start><S{self.config.scale}>"
        return self.tokenizer(input_text, return_tensors="pt")

    def generate_image(self, prompt: str) -> Image.Image:
        inputs = self._prompt_to_input_ids(prompt)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        gen_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )

        gen_text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=False)[0]
        gen_code = [int(x) for x in re.findall(r"<I(\d+)>", gen_text)]
        gen_code = gen_code[: self.config.seq_len] + [0] * max(0, self.config.seq_len - len(gen_code))
        gen_code = torch.tensor(gen_code, dtype=torch.long, device=self.device).unsqueeze(0)

        gen_tensor = self.visual_tokenizer.decode_from_encoder_indices(
            gen_code,
            {
                "cfg_scale": self.config.cfg_scale,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "guidance_scale_pow": self.config.guidance_scale_pow,
            },
        )
        return Image.fromarray(gen_tensor[0].numpy())


def parse_args():
    parser = argparse.ArgumentParser(description="Text-to-image inference with Qwen/Tar + RAR-DTok.")
    parser.add_argument("--model-path", type=str, default=T2IRARConfig.model_path)
    parser.add_argument("--rar-path", type=str, default=T2IRARConfig.rar_path)
    parser.add_argument("--encoder-path", type=str, default=T2IRARConfig.encoder_path)
    parser.add_argument("--decoder-path", type=str, default=T2IRARConfig.decoder_path)
    parser.add_argument("--prompt", type=str, default="A photo of a macaw")
    parser.add_argument("--output", type=str, default="generated_rar.png")
    parser.add_argument("--device", type=str, default=T2IRARConfig.device)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--scale", type=int, default=T2IRARConfig.scale)
    parser.add_argument("--seq-len", type=int, default=T2IRARConfig.seq_len)
    parser.add_argument("--temperature", type=float, default=T2IRARConfig.temperature)
    parser.add_argument("--top-p", type=float, default=T2IRARConfig.top_p)
    parser.add_argument("--top-k", type=int, default=T2IRARConfig.top_k)
    parser.add_argument("--cfg-scale", type=float, default=T2IRARConfig.cfg_scale)
    parser.add_argument("--guidance-scale-pow", type=float, default=T2IRARConfig.guidance_scale_pow)
    return parser.parse_args()


def parse_dtype(name):
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    return torch.bfloat16


def main():
    args = parse_args()
    config = T2IRARConfig(
        model_path=args.model_path,
        rar_path=args.rar_path,
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        scale=args.scale,
        seq_len=args.seq_len,
        max_new_tokens=args.seq_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        cfg_scale=args.cfg_scale,
        guidance_scale_pow=args.guidance_scale_pow,
    )
    inference = TextToImageRARInference(config)
    image = inference.generate_image(args.prompt)
    image.save(args.output)
    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
