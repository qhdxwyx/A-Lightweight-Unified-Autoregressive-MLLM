from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms.functional import to_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

import llava.model  # noqa: F401
from llava.utils import apply_chat_template
from tok.ta_tok import TextAlignedTokenizer


@dataclass
class I2TConfig:
    model_path: str = "csuhan/Tar-1.5B"
    ta_tok_path: str = "ta_tok.pth"
    device: str = "cuda:0"
    max_new_tokens: int = 256

class ImageToTextInference:
    def __init__(self, config: I2TConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_path).to(self.device)
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.model_path, fix_mistral_regex=True)
        self.visual_tokenizer = TextAlignedTokenizer.from_checkpoint(
            config.ta_tok_path, load_teacher=False, input_type='indices'
        ).to(self.device)

    def generate(self, image_path: str, prompt: str) -> str:
        image = Image.open(image_path).convert('RGB')
        image = to_tensor(image).unsqueeze(0).to(self.device)
        
        image_code = self.visual_tokenizer(image)['encoded']
        image_text = "".join([f"<I{x}>" for x in image_code[0].cpu().tolist()])
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{image_text}\n{prompt}"}
        ]

        input_text = apply_chat_template(self.text_tokenizer, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.text_tokenizer(input_text, return_tensors="pt")
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        gen_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True)
        if gen_ids.shape[1] > input_ids.shape[1]:
            gen_ids = gen_ids[:, input_ids.shape[1]:]
        return self.text_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

if __name__ == "__main__":
    config = I2TConfig()
    config.ta_tok_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
    inference = ImageToTextInference(config)
    description = inference.generate('asset/dog_cat.jpg', "Describe the image shortly.")
    print(description)
