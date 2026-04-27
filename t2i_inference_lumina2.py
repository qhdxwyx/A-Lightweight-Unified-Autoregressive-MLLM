import re
from dataclasses import dataclass
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

import llava.model  # noqa: F401
from llava.utils import apply_chat_template
from tok.dif_dtok_lumina2 import LuminaAutoEncoder


@dataclass
class T2IConfig:
    model_path: str = "csuhan/Tar-1.5B"
    # visual tokenizer config
    lumina2_path: str = "csuhan/Tar-Lumina2"
    ta_tok_path: str = "ta_tok.pth"
    text_encoder: str ="google/gemma-2-2b"
    vae_path="black-forest-labs/FLUX.1-dev"
    num_sampling_steps=50
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    # generation parameters
    scale: int = 0  # choose from [0, 1, 2]
    seq_len: int = 729  # choose from [729, 169, 81]
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 1200


class TextToImageInference:
    def __init__(self, config: T2IConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.set_grad_enabled(False)
        self._load_models()

    def _load_models(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, torch_dtype=self.config.dtype
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        visual_tokenizer_cfg = dict(
            ckpt=self.config.lumina2_path,
            ta_tok_path=self.config.ta_tok_path,
            text_encoder=self.config.text_encoder,
            vae_path=self.config.vae_path,
            resolution=1024,
            num_sampling_steps=self.config.num_sampling_steps,
            solver="euler",
            num_gpus=1,
            ema=True,
            time_shifting_factor=6.0,
            prompt="You are an assistant designed to generate superior images with the highest degree of image alignment based on the SigLIP2 features of an original image. <Prompt Start> ",
        )
        self.visual_tokenizer = LuminaAutoEncoder(
            visual_tokenizer_cfg, device=self.device, dtype=self.config.dtype)

    def generate_image(self, prompt: str) -> Image.Image:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        input_text = apply_chat_template(self.tokenizer, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_text += f"<im_start><S{self.config.scale}>"

        inputs = self.tokenizer(input_text, return_tensors="pt")

        gen_ids = self.model.generate(
            inputs.input_ids.to(self.device),
            max_new_tokens=self.config.seq_len,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k
        )

        gen_text = self.tokenizer.batch_decode(gen_ids)[0]
        gen_code = [int(x) for x in re.findall(r'<I(\d+)>', gen_text)]
        gen_code = gen_code[:self.config.seq_len] + [0] * max(0, self.config.seq_len - len(gen_code))
        gen_code = torch.tensor(gen_code).unsqueeze(0).to(self.device)
        with torch.autocast("cuda", torch.bfloat16):
            out_image = self.visual_tokenizer.infer(indices=gen_code)
        return out_image


def main():
    config = T2IConfig()
    config.lumina2_path = snapshot_download("csuhan/Tar-Lumina2")
    config.ta_tok_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
    inference = TextToImageInference(config)

    prompt = "A photo of a macaw"
    image = inference.generate_image(prompt)
    image.save("generated_image.png")


if __name__ == "__main__":
    main()
