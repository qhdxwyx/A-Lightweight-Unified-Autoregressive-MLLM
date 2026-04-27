import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from scripts.train_rar_dtok import ImageOnlyDataset, ParquetImageDataset
from tok.ar_dtok.vqvae import VQVAE
from tok.rar_dtok.rar_model import RARCondModel
from tok.ta_tok import TextAlignedTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained RAR-DTok checkpoint.")
    parser.add_argument("--rar-path", type=str, default="output_dir_RAR_DTok/latest.pth")
    parser.add_argument("--ta-tok-path", type=str, default="assets/common/ta_tok.pth")
    parser.add_argument("--vqvae-path", type=str, default="assets/common/vq_ds16_t2i.pt")
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--parquet-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="eval_outputs/rar_dtok")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--pool-scale", type=int, default=1)
    parser.add_argument(
        "--save-samples",
        type=int,
        default=-1,
        help="Number of original/reconstruction pairs to save. Use -1 to save all evaluated images, 0 to disable.",
    )
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_dtype(name, device):
    if name == "fp32" or device.type == "cpu":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def post_process(x):
    return x.float().clamp(0.0, 1.0)


def build_cfg(args):
    return {
        "data": {
            "image_root": args.image_root,
            "manifest": args.manifest,
            "parquet_root": args.parquet_root,
            "recursive": True,
            "random_crop": False,
            "random_flip": False,
            "resize_shorter_edge": 256,
        },
        "model": {
            "image_size": 256,
        },
    }


def build_dataset(args):
    cfg = build_cfg(args)
    if args.parquet_root:
        dataset = ParquetImageDataset(cfg)
    else:
        dataset = ImageOnlyDataset(cfg)

    if args.limit and not isinstance(dataset, torch.utils.data.IterableDataset):
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    return dataset


def get_sample_source(dataset, sample_index):
    if isinstance(dataset, Subset):
        original_index = dataset.indices[sample_index]
        return get_sample_source(dataset.dataset, original_index)
    paths = getattr(dataset, "paths", None)
    if paths is not None and sample_index < len(paths):
        return str(paths[sample_index])
    return None


@torch.inference_mode()
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.image_root is None and args.manifest is None and args.parquet_root is None:
        raise ValueError("Set --image-root, --manifest, or --parquet-root.")

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = resolve_dtype(args.dtype, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading RAR-DTok: {args.rar_path}")
    rar = RARCondModel.from_checkpoint(args.rar_path).eval().to(device=device, dtype=dtype)

    print(f"Loading TA-Tok: {args.ta_tok_path}")
    ta_tok = TextAlignedTokenizer.from_checkpoint(
        args.ta_tok_path,
        load_teacher=False,
        input_type="rec",
    ).eval().to(device=device, dtype=dtype)
    if hasattr(ta_tok.bottleneck, "regularizer") and hasattr(ta_tok.bottleneck.regularizer, "set_eval_deterministic"):
        ta_tok.bottleneck.regularizer.set_eval_deterministic(True)

    print(f"Loading VQ-VAE: {args.vqvae_path}")
    vqvae = VQVAE.from_checkpoint(args.vqvae_path).eval().to(device=device)

    dataset = build_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    total_images = 0
    saved = 0
    save_all_samples = args.save_samples < 0
    seen = 0
    sample_records = []

    pbar = tqdm(dataloader, desc="Evaluating RAR-DTok")
    for images in pbar:
        images = images.to(device, non_blocking=True)

        condition = ta_tok(images.to(dtype=dtype), pool_scale=args.pool_scale)["encoded"]
        target_ids = vqvae.encode_to_indices(images).long()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
            logits, labels = rar(target_ids, condition, return_labels=True)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        predictions = torch.argmax(logits, dim=-1)
        token_acc = (predictions == labels).float().mean()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_acc += token_acc.item() * batch_size
        total_batches += 1
        total_images += batch_size
        pbar.set_postfix(loss=total_loss / total_images, token_acc=total_acc / total_images)

        if save_all_samples or saved < args.save_samples:
            sample_n = batch_size if save_all_samples else min(batch_size, args.save_samples - saved)
            sample_cond = condition[:sample_n]
            sample_ids = rar.sample(
                sample_cond,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seq_length=rar.image_seq_len,
            )
            rec = post_process(vqvae.decode_from_bottleneck(sample_ids))
            originals = post_process(images[:sample_n])
            for i in range(sample_n):
                sample_id = saved + i
                pair = torch.stack([originals[i], rec[i]], dim=0)
                file_name = f"sample_{sample_id:04d}_orig_recon.png"
                save_image(pair, output_dir / file_name, nrow=2)
                dataset_index = seen + i
                sample_records.append(
                    {
                        "sample_id": sample_id,
                        "file": file_name,
                        "dataset_index": dataset_index,
                        "source": get_sample_source(dataset, dataset_index),
                    }
                )
            saved += sample_n
        seen += batch_size

    metrics = {
        "num_images": total_images,
        "num_batches": total_batches,
        "loss": total_loss / max(total_images, 1),
        "perplexity": math.exp(total_loss / max(total_images, 1)),
        "token_acc": total_acc / max(total_images, 1),
        "pool_scale": args.pool_scale,
        "rar_path": str(args.rar_path),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for record in sample_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")
    if saved:
        print(f"Saved {saved} original/reconstruction pairs to {output_dir}")
        print(f"Saved sample metadata to {samples_path}")


if __name__ == "__main__":
    main()
