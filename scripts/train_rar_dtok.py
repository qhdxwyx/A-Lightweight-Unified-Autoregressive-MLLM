import argparse
import io
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image, ImageFile
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
import pyarrow.parquet as pq

try:
    import wandb
except ImportError:
    wandb = None

from tok.ar_dtok.vqvae import VQVAE
from tok.rar_dtok.rar_model import RARCondModel
from tok.ta_tok import TextAlignedTokenizer
from tok.utils import load_torch_checkpoint

ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Tar De-Tokenizer with a conditional RAR-L model.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path.")
    parser.add_argument("--image-root", type=str, default=None, help="Override data.image_root from config.")
    parser.add_argument("--manifest", type=str, default=None, help="Override data.manifest from config.")
    parser.add_argument("--parquet-root", type=str, default=None, help="Override data.parquet_root from config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir from config.")
    parser.add_argument("--ta-tok-path", type=str, default=None, help="Override tokenizer.ta_tok_path from config.")
    parser.add_argument("--vqvae-path", type=str, default=None, help="Override tokenizer.vqvae_path from config.")
    parser.add_argument("--init-from", type=str, default=None, help="Optional checkpoint to initialize from.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional train.batch_size override.")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Optional train.grad_accum_steps override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional train.num_workers override.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Optional train.prefetch_factor override.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Optional train.warmup_steps override.")
    parser.add_argument("--warmup-ratio", type=float, default=None, help="Optional train.warmup_ratio override. Takes precedence over warmup_steps.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max_steps override.")
    parser.add_argument("--start-step", type=int, default=None, help="Optional global step to continue schedules and logging from.")
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("data", {})
    cfg.setdefault("tokenizer", {})
    cfg.setdefault("train", {})
    cfg.setdefault("model", {})

    if args.image_root is not None:
        cfg["data"]["image_root"] = args.image_root
    if args.manifest is not None:
        cfg["data"]["manifest"] = args.manifest
    if args.parquet_root is not None:
        cfg["data"]["parquet_root"] = args.parquet_root
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.ta_tok_path is not None:
        cfg["tokenizer"]["ta_tok_path"] = args.ta_tok_path
    if args.vqvae_path is not None:
        cfg["tokenizer"]["vqvae_path"] = args.vqvae_path
    if args.init_from is not None:
        cfg["train"]["init_from"] = args.init_from
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        cfg["train"]["grad_accum_steps"] = args.grad_accum_steps
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.prefetch_factor is not None:
        cfg["train"]["prefetch_factor"] = args.prefetch_factor
    if args.warmup_steps is not None:
        cfg["train"]["warmup_steps"] = args.warmup_steps
    if args.warmup_ratio is not None:
        cfg["train"]["warmup_ratio"] = args.warmup_ratio
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.start_step is not None:
        cfg["train"]["start_step"] = args.start_step

    cfg["output_dir"] = cfg.get("output_dir", "output_dir/rar_l_dtok")
    cfg["train"]["pool_scales"] = cfg["train"].get("pool_scales", [1, 2, 3])
    cfg["train"]["pool_scale_weights"] = cfg["train"].get("pool_scale_weights", [2, 1, 1])
    cfg["train"]["mixed_precision"] = cfg["train"].get("mixed_precision", "bf16")
    cfg["train"]["learning_rate"] = cfg["train"].get("learning_rate", 4e-4)
    cfg["train"]["weight_decay"] = cfg["train"].get("weight_decay", 0.03)
    cfg["train"]["adam_beta1"] = cfg["train"].get("adam_beta1", 0.9)
    cfg["train"]["adam_beta2"] = cfg["train"].get("adam_beta2", 0.96)
    cfg["train"]["max_grad_norm"] = cfg["train"].get("max_grad_norm", 1.0)
    cfg["train"]["batch_size"] = cfg["train"].get("batch_size", 8)
    cfg["train"]["grad_accum_steps"] = cfg["train"].get("grad_accum_steps", 1)
    cfg["train"]["num_workers"] = cfg["train"].get("num_workers", 4)
    cfg["train"]["prefetch_factor"] = cfg["train"].get("prefetch_factor", 2)
    cfg["train"]["log_every"] = cfg["train"].get("log_every", 20)
    cfg["train"]["save_every"] = cfg["train"].get("save_every", 5000)
    cfg["train"]["seed"] = cfg["train"].get("seed", 42)
    cfg["train"]["max_steps"] = cfg["train"].get("max_steps", 100000)
    cfg["train"]["warmup_ratio"] = cfg["train"].get("warmup_ratio", None)
    if cfg["train"]["warmup_ratio"] is not None:
        cfg["train"]["warmup_steps"] = int(cfg["train"]["max_steps"] * float(cfg["train"]["warmup_ratio"]))
    else:
        cfg["train"]["warmup_steps"] = cfg["train"].get("warmup_steps", 2000)
    cfg["train"]["random_ratio_start"] = cfg["train"].get("random_ratio_start", 1.0)
    cfg["train"]["random_ratio_end"] = cfg["train"].get("random_ratio_end", 0.0)
    cfg["train"]["random_ratio_anneal_start"] = cfg["train"].get("random_ratio_anneal_start", 0)
    cfg["train"]["random_ratio_anneal_end"] = cfg["train"].get("random_ratio_anneal_end", cfg["train"]["max_steps"])
    cfg["train"]["start_step"] = int(cfg["train"].get("start_step", 0))
    cfg["data"]["recursive"] = cfg["data"].get("recursive", True)
    cfg["data"]["random_crop"] = cfg["data"].get("random_crop", False)
    cfg["data"]["random_flip"] = cfg["data"].get("random_flip", True)
    cfg["data"]["parquet_root"] = cfg["data"].get("parquet_root", None)

    image_size = cfg["model"].get("image_size", 256)
    cfg["model"]["image_size"] = image_size
    cfg["model"]["image_seq_len"] = cfg["model"].get("image_seq_len", (image_size // 16) ** 2)
    cfg["model"]["max_cond_len"] = cfg["model"].get("max_cond_len", 729)
    cfg["model"]["name"] = cfg["model"].get("name", "rar-cond-L")
    cfg["model"]["vocab_size"] = cfg["model"].get("vocab_size", 16384)
    cfg["model"]["cond_dim"] = cfg["model"].get("cond_dim", 1152)
    cfg["model"]["dropout"] = cfg["model"].get("dropout", 0.1)
    cfg["model"]["attn_drop"] = cfg["model"].get("attn_drop", 0.1)
    cfg["model"]["cond_dropout_prob"] = cfg["model"].get("cond_dropout_prob", 0.1)
    cfg["model"]["use_checkpoint"] = cfg["model"].get("use_checkpoint", False)

    return cfg


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if manifest_path.suffix in {".txt", ".lst"}:
        return [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if manifest_path.suffix == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = []
        for item in payload:
            if isinstance(item, str):
                paths.append(item)
            elif isinstance(item, dict):
                paths.append(item["image"])
        return paths
    if manifest_path.suffix == ".jsonl":
        paths = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            paths.append(item if isinstance(item, str) else item["image"])
        return paths
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def scan_images(image_root, recursive=True):
    image_root = Path(image_root)
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    pattern = "**/*" if recursive else "*"
    return [str(path) for path in sorted(image_root.glob(pattern)) if path.suffix.lower() in IMAGE_EXTENSIONS]


def scan_parquet_files(parquet_root, recursive=True):
    parquet_root = Path(parquet_root)
    if not parquet_root.exists():
        raise FileNotFoundError(f"Parquet root not found: {parquet_root}")
    pattern = "**/*.parquet" if recursive else "*.parquet"
    return [str(path) for path in sorted(parquet_root.glob(pattern))]


def decode_parquet_image(image_field):
    if image_field is None:
        raise ValueError("Parquet sample does not contain an image.")
    if isinstance(image_field, dict) and "bytes" in image_field:
        image_bytes = image_field["bytes"]
    elif isinstance(image_field, (bytes, bytearray)):
        image_bytes = bytes(image_field)
    elif isinstance(image_field, list) and image_field:
        first = image_field[0]
        if isinstance(first, dict) and "bytes" in first:
            image_bytes = first["bytes"]
        else:
            raise ValueError("Unsupported image list format in parquet dataset.")
    else:
        raise ValueError(f"Unsupported parquet image format: {type(image_field)}")
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


class ImageOnlyDataset(Dataset):
    def __init__(self, cfg):
        manifest = cfg["data"].get("manifest")
        image_root = cfg["data"].get("image_root")

        if manifest:
            self.paths = load_manifest(manifest)
            if image_root:
                self.paths = [str((Path(image_root) / path).resolve()) if not Path(path).is_absolute() else path for path in self.paths]
        elif image_root:
            self.paths = scan_images(image_root, recursive=cfg["data"]["recursive"])
        else:
            raise ValueError("Either data.image_root or data.manifest must be provided.")

        if not self.paths:
            raise ValueError("No training images were found.")

        crop_size = cfg["model"]["image_size"]
        resize_size = cfg["data"].get("resize_shorter_edge", crop_size)
        transform_steps = [transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)]
        if cfg["data"]["random_crop"]:
            transform_steps.append(transforms.RandomCrop(crop_size))
        else:
            transform_steps.append(transforms.CenterCrop(crop_size))
        if cfg["data"]["random_flip"]:
            transform_steps.append(transforms.RandomHorizontalFlip())
        transform_steps.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_steps)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image)


class ParquetImageDataset(IterableDataset):
    def __init__(self, cfg):
        parquet_root = cfg["data"].get("parquet_root")
        if not parquet_root:
            raise ValueError("data.parquet_root must be provided for parquet training.")

        self.paths = scan_parquet_files(parquet_root, recursive=cfg["data"]["recursive"])
        if not self.paths:
            raise ValueError("No parquet files were found.")

        crop_size = cfg["model"]["image_size"]
        resize_size = cfg["data"].get("resize_shorter_edge", crop_size)
        transform_steps = [transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)]
        if cfg["data"]["random_crop"]:
            transform_steps.append(transforms.RandomCrop(crop_size))
        else:
            transform_steps.append(transforms.CenterCrop(crop_size))
        if cfg["data"]["random_flip"]:
            transform_steps.append(transforms.RandomHorizontalFlip())
        transform_steps.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_steps)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        paths = list(self.paths)
        if len(paths) < total_workers:
            paths.extend(random.choices(paths, k=total_workers - len(paths)))
        files_iter = paths[global_worker_id::total_workers]

        random.shuffle(files_iter)
        for file_path in files_iter:
            parquet_file = pq.ParquetFile(file_path)
            for rg in range(parquet_file.num_row_groups):
                rows = parquet_file.read_row_group(rg, columns=["image"]).to_pylist()
                for row in rows:
                    try:
                        image = decode_parquet_image(row["image"])
                        yield self.transform(image)
                    except Exception:
                        continue


def unwrap_state_dict(checkpoint):
    if "model" in checkpoint and isinstance(checkpoint["model"], dict) and "sd" in checkpoint["model"]:
        return checkpoint["model"]["sd"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def maybe_resize_sequence_parameter(tensor, target_shape):
    if tuple(tensor.shape) == tuple(target_shape):
        return tensor
    if tensor.ndim == 3 and len(target_shape) == 3 and tensor.shape[0] == target_shape[0] and tensor.shape[2] == target_shape[2]:
        resized = F.interpolate(tensor.transpose(1, 2), size=target_shape[1], mode="linear", align_corners=False).transpose(1, 2)
        return resized
    return None


def load_model_checkpoint(model, checkpoint_path):
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)
    model_state = model.state_dict()
    new_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if tuple(value.shape) == tuple(model_state[key].shape):
            new_state[key] = value
            continue
        resized = maybe_resize_sequence_parameter(value, model_state[key].shape)
        if resized is not None:
            new_state[key] = resized.to(dtype=model_state[key].dtype)
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    return missing, unexpected, skipped


def build_model_args(cfg):
    return {
        "cond_dim": cfg["model"]["cond_dim"],
        "max_cond_len": cfg["model"]["max_cond_len"],
        "image_seq_len": cfg["model"]["image_seq_len"],
        "vocab_size": cfg["model"]["vocab_size"],
        "dropout": cfg["model"]["dropout"],
        "attn_drop": cfg["model"]["attn_drop"],
        "cond_dropout_prob": cfg["model"]["cond_dropout_prob"],
        "use_checkpoint": cfg["model"]["use_checkpoint"],
    }


def build_model(cfg):
    name = cfg["model"]["name"]
    model_args = build_model_args(cfg)
    if name == "rar-cond-L":
        return RARCondModel.from_checkpoint({"model": {"name": name, "args": model_args, "sd": {}}}, load_state_dict=False)
    if name == "rar-cond-B":
        return RARCondModel.from_checkpoint({"model": {"name": name, "args": model_args, "sd": {}}}, load_state_dict=False)
    if name == "rar-cond-XL":
        return RARCondModel.from_checkpoint({"model": {"name": name, "args": model_args, "sd": {}}}, load_state_dict=False)
    raise ValueError(f"Unsupported model name: {name}")


def build_checkpoint(model, cfg, state_dict):
    return {
        "model": {
            "name": cfg["model"]["name"],
            "args": build_model_args(cfg),
            "sd": state_dict,
        },
        "meta": {
            "image_size": cfg["model"]["image_size"],
            "pool_scales": cfg["train"]["pool_scales"],
            "ta_tok_path": cfg["tokenizer"]["ta_tok_path"],
            "vqvae_path": cfg["tokenizer"]["vqvae_path"],
        },
    }


def get_random_ratio(cfg, step):
    start = cfg["train"]["random_ratio_start"]
    end = cfg["train"]["random_ratio_end"]
    anneal_start = cfg["train"]["random_ratio_anneal_start"]
    anneal_end = cfg["train"]["random_ratio_anneal_end"]

    if step <= anneal_start:
        return start
    if step >= anneal_end:
        return end
    progress = (step - anneal_start) / max(anneal_end - anneal_start, 1)
    return start + progress * (end - start)


def align_scheduler_to_step(scheduler, step):
    if step <= 0:
        return
    if not hasattr(scheduler, "lr_lambdas"):
        raise TypeError("Cannot align this scheduler type to a non-zero start_step.")

    scheduler.last_epoch = step
    last_lrs = []
    for param_group, lr_lambda, base_lr in zip(
        scheduler.optimizer.param_groups,
        scheduler.lr_lambdas,
        scheduler.base_lrs,
    ):
        lr = base_lr * lr_lambda(step)
        param_group["lr"] = lr
        last_lrs.append(lr)
    scheduler._last_lr = last_lrs


def save_checkpoint(accelerator, model, cfg, step):
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = accelerator.get_state_dict(model)
    checkpoint = build_checkpoint(accelerator.unwrap_model(model), cfg, state_dict)
    checkpoint_path = output_dir / f"checkpoint-{step:07d}.pth"
    accelerator.save(checkpoint, checkpoint_path)
    if accelerator.is_main_process:
        latest_path = output_dir / "latest.pth"
        accelerator.save(checkpoint, latest_path)


def main():
    args = parse_args()
    cfg = load_config(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["train"]["grad_accum_steps"],
        mixed_precision=cfg["train"]["mixed_precision"],
    )

    if accelerator.is_main_process:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        with open(Path(cfg["output_dir"]) / "train_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    use_wandb = (
        accelerator.is_main_process
        and wandb is not None
        and os.environ.get("WANDB_MODE", "").lower() != "disabled"
    )
    if accelerator.is_main_process and wandb is None and (
        os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_PROJECT") or os.environ.get("WANDB_RUN_NAME")
    ):
        print("wandb is not installed; skip online logging for RAR-L training.")

    if use_wandb:
        try:
            api_key = os.environ.get("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key, relogin=False)
            elif not getattr(getattr(wandb, "api", None), "api_key", None):
                print("No WANDB_API_KEY found and no existing wandb login detected; skip online logging for RAR-L training.")
                use_wandb = False
        except Exception as exc:
            print(f"wandb authentication unavailable ({exc}); skip online logging for RAR-L training.")
            use_wandb = False

    if use_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "tar-rar-dtok"),
            name=os.environ.get("WANDB_RUN_NAME", Path(cfg["output_dir"]).name),
            config=cfg,
            dir=cfg["output_dir"],
        )

    set_seed(cfg["train"]["seed"], device_specific=True)

    if cfg["data"].get("parquet_root"):
        dataset = ParquetImageDataset(cfg)
    else:
        dataset = ImageOnlyDataset(cfg)
    dataloader_kwargs = {
        "batch_size": cfg["train"]["batch_size"],
        "shuffle": not isinstance(dataset, IterableDataset),
        "num_workers": cfg["train"]["num_workers"],
        "pin_memory": True,
        "drop_last": not isinstance(dataset, IterableDataset),
        "persistent_workers": cfg["train"]["num_workers"] > 0,
    }
    if cfg["train"]["num_workers"] > 0:
        dataloader_kwargs["prefetch_factor"] = cfg["train"]["prefetch_factor"]
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    model = build_model(cfg)
    if cfg["train"].get("init_from"):
        missing, unexpected, skipped = load_model_checkpoint(model, cfg["train"]["init_from"])
        if accelerator.is_main_process:
            print(f"Loaded init checkpoint {cfg['train']['init_from']}")
            print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}, resized/skipped keys: {skipped}")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        betas=(cfg["train"]["adam_beta1"], cfg["train"]["adam_beta2"]),
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["train"]["warmup_steps"],
        num_training_steps=cfg["train"]["max_steps"],
    )
    start_step = cfg["train"]["start_step"]
    align_scheduler_to_step(scheduler, start_step)
    if accelerator.is_main_process and start_step > 0:
        print(
            f"Starting from global_step={start_step}; "
            f"training until max_steps={cfg['train']['max_steps']}."
        )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    ta_tok = TextAlignedTokenizer.from_checkpoint(
        cfg["tokenizer"]["ta_tok_path"],
        load_teacher=False,
        input_type="rec",
    ).to(accelerator.device)
    if hasattr(ta_tok.bottleneck, "regularizer") and hasattr(ta_tok.bottleneck.regularizer, "set_eval_deterministic"):
        ta_tok.bottleneck.regularizer.set_eval_deterministic(True)
    ta_tok.eval()
    ta_tok.requires_grad_(False)

    vqvae = VQVAE.from_checkpoint(cfg["tokenizer"]["vqvae_path"]).to(accelerator.device)
    vqvae.eval()
    vqvae.requires_grad_(False)

    global_step = start_step
    running_loss = 0.0
    running_acc = 0.0
    step_count = 0

    while global_step < cfg["train"]["max_steps"]:
        for images in dataloader:
            with accelerator.accumulate(model):
                pool_scale = random.choices(
                    cfg["train"]["pool_scales"],
                    weights=cfg["train"]["pool_scale_weights"],
                    k=1,
                )[0]
                random_ratio = get_random_ratio(cfg, global_step)
                accelerator.unwrap_model(model).set_random_ratio(random_ratio)

                with torch.no_grad():
                    condition = ta_tok(images, pool_scale=pool_scale)["encoded"]
                    target_ids = vqvae.encode_to_indices(images).long()

                with accelerator.autocast():
                    logits, labels = model(target_ids, condition, return_labels=True)
                    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg["train"]["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                token_acc = (predictions == labels).float().mean()
                running_loss += accelerator.gather(loss.detach()).mean().item()
                running_acc += accelerator.gather(token_acc.detach()).mean().item()
                step_count += 1

            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process and global_step % cfg["train"]["log_every"] == 0:
                    avg_loss = running_loss / max(step_count, 1)
                    avg_acc = running_acc / max(step_count, 1)
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"step={global_step} "
                        f"loss={avg_loss:.4f} "
                        f"token_acc={avg_acc:.4f} "
                        f"pool_scale={pool_scale} "
                        f"random_ratio={random_ratio:.4f} "
                        f"lr={lr:.6e}"
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "train/step": global_step,
                                "train/loss": avg_loss,
                                "train/token_acc": avg_acc,
                                "train/lr": lr,
                                "train/pool_scale": pool_scale,
                                "train/random_ratio": random_ratio,
                            },
                            step=global_step,
                        )
                    running_loss = 0.0
                    running_acc = 0.0
                    step_count = 0

                if global_step % cfg["train"]["save_every"] == 0:
                    accelerator.wait_for_everyone()
                    save_checkpoint(accelerator, model, cfg, global_step)

                if global_step >= cfg["train"]["max_steps"]:
                    break

    accelerator.wait_for_everyone()
    save_checkpoint(accelerator, model, cfg, global_step)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
