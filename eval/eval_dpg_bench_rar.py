import argparse
import glob
import os
import re

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
from tok.rar_autoencoder import RARMMAutoEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="../dpg_bench/prompts")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--rar_path", type=str, required=True)
    parser.add_argument("--encoder_path", type=str, default="/tmp/ta_tok.pth")
    parser.add_argument("--decoder_path", type=str, default="/tmp/vq_ds16_t2i.pt")
    parser.add_argument("--seq_len", type=int, default=729)
    parser.add_argument("--seq_scale", type=int, default=1)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    return parser.parse_args()


def load_visual_tokenizer(args):
    visual_tokenizer = RARMMAutoEncoder(
        rar_path=args.rar_path,
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path,
        encoder_args={"input_type": "rec"},
        decoder_args={},
    ).eval()
    return visual_tokenizer


def get_prompt_template(args):
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<im_start>"
        + f"<S{str(args.seq_scale - 1)}>"
    )


class DPGBenchDataset(Dataset):
    def __init__(self, args, tokenizer):
        prompt_files = glob.glob(os.path.join(args.prompts, "*.txt"))
        self.prompt_names = [os.path.basename(x).split(".")[0] for x in prompt_files]
        self.prompts = [open(x, encoding="utf-8").read().strip() for x in prompt_files]

        prompt_temp = get_prompt_template(args)
        self.all_input_ids = []
        for prompt in self.prompts:
            question = prompt_temp.format(prompt)
            input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")[None]
            self.all_input_ids.extend(input_ids)

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.prompt_names[idx]


if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    args = parse_args()
    visual_tokenizer = load_visual_tokenizer(args).to(device=device, dtype=dtype)

    tokenizer, model, _, _ = load_pretrained_model(
        args.model,
        None,
        "llava_qwen",
        device_map=device,
        multimodal=True,
        attn_implementation="sdpa",
    )
    model.eval().to(device=device, dtype=dtype)

    dataset = DPGBenchDataset(args, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    seq_len = args.seq_len
    os.makedirs(args.save_dir, exist_ok=True)
    for data in tqdm(dataloader):
        input_ids, file_names = data
        input_ids = torch.cat([input_ids] * args.repeat)
        with autocast(dtype=model.dtype):
            cont = model.generate(
                input_ids.to(device),
                images=None,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=seq_len,
            )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)

        codes = []
        for text_output in text_outputs:
            code = [int(x) for x in re.findall(r"<I(\d+)>", text_output)]
            code = code[:seq_len] + [0] * max(0, seq_len - len(code))
            codes.append(code)
        codes = torch.tensor(codes, dtype=torch.long, device=device)[:, :seq_len]

        with torch.no_grad():
            recs = visual_tokenizer.decode_from_encoder_indices(codes, {"cfg_scale": args.cfg_scale})
            recs = recs.numpy()

        top_row = np.concatenate((recs[0], recs[1]), axis=1)
        bottom_row = np.concatenate((recs[2], recs[3]), axis=1)
        final_image = np.concatenate((top_row, bottom_row), axis=0)
        save_path = os.path.join(args.save_dir, file_names[0] + ".png")
        Image.fromarray(final_image).save(save_path)

    dist.barrier()
    dist.destroy_process_group()
