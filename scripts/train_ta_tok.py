import argparse
import io
import json
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
from transformers import AutoModel, get_cosine_schedule_with_warmup

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    import wandb
except ImportError:
    wandb = None

from tok.ta_tok import TextAlignedTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Text-Aligned Tokenizer.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path.")
    parser.add_argument("--image-root", type=str, default=None, help="Override data.image_root from config.")
    parser.add_argument("--manifest", type=str, default=None, help="Override data.manifest from config.")
    parser.add_argument("--parquet-root", type=str, default=None, help="Override data.parquet_root from config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir from config.")
    parser.add_argument("--init-from", type=str, default=None, help="Optional Text-Aligned Tokenizer checkpoint.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional train.batch_size override.")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Optional train.grad_accum_steps override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional train.num_workers override.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Optional train.prefetch_factor override.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Optional train.warmup_steps override.")
    parser.add_argument("--warmup-ratio", type=float, default=None, help="Optional train.warmup_ratio override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional train.max_steps override.")
    parser.add_argument("--start-step", type=int, default=None, help="Optional global step for resumed schedules.")
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})

    if args.image_root is not None:
        cfg["data"]["image_root"] = args.image_root
    if args.manifest is not None:
        cfg["data"]["manifest"] = args.manifest
    if args.parquet_root is not None:
        cfg["data"]["parquet_root"] = args.parquet_root
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
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

    cfg["output_dir"] = cfg.get("output_dir", "output_dir/ta_tok_train")

    cfg["data"]["image_root"] = cfg["data"].get("image_root", None)
    cfg["data"]["manifest"] = cfg["data"].get("manifest", None)
    cfg["data"]["parquet_root"] = cfg["data"].get("parquet_root", None)
    cfg["data"]["recursive"] = cfg["data"].get("recursive", True)
    cfg["data"]["random_crop"] = cfg["data"].get("random_crop", False)
    cfg["data"]["random_flip"] = cfg["data"].get("random_flip", True)

    cfg["model"]["input_size"] = cfg["model"].get("input_size", 384)
    cfg["model"]["image_size"] = cfg["model"].get("image_size", cfg["model"]["input_size"])
    cfg["model"]["resize_shorter_edge"] = cfg["model"].get("resize_shorter_edge", cfg["model"]["input_size"])
    cfg["model"]["teacher"] = cfg["model"].get("teacher", "google/siglip2-so400m-patch14-384")
    cfg["model"]["select_layer_id"] = cfg["model"].get("select_layer_id", -2)
    cfg["model"]["bottleneck_token_num"] = cfg["model"].get("bottleneck_token_num", 729)
    cfg["model"]["pool_scales"] = cfg["model"].get("pool_scales", [1])
    cfg["model"]["pool_scale_weights"] = cfg["model"].get("pool_scale_weights", [1] * len(cfg["model"]["pool_scales"]))
    cfg["model"]["decoder_depth"] = cfg["model"].get("decoder_depth", 3)

    cfg["train"]["seed"] = cfg["train"].get("seed", 42)
    cfg["train"]["batch_size"] = cfg["train"].get("batch_size", 2)
    cfg["train"]["grad_accum_steps"] = cfg["train"].get("grad_accum_steps", 8)
    cfg["train"]["num_workers"] = cfg["train"].get("num_workers", 4)
    cfg["train"]["prefetch_factor"] = cfg["train"].get("prefetch_factor", 2)
    cfg["train"]["mixed_precision"] = cfg["train"].get("mixed_precision", "bf16")
    cfg["train"]["learning_rate"] = cfg["train"].get("learning_rate", 1e-4)
    cfg["train"]["weight_decay"] = cfg["train"].get("weight_decay", 0.03)
    cfg["train"]["adam_beta1"] = cfg["train"].get("adam_beta1", 0.9)
    cfg["train"]["adam_beta2"] = cfg["train"].get("adam_beta2", 0.95)
    cfg["train"]["max_steps"] = cfg["train"].get("max_steps", 100000)
    cfg["train"]["warmup_ratio"] = cfg["train"].get("warmup_ratio", None)
    if cfg["train"]["warmup_ratio"] is not None:
        cfg["train"]["warmup_steps"] = int(cfg["train"]["max_steps"] * float(cfg["train"]["warmup_ratio"]))
    else:
        cfg["train"]["warmup_steps"] = cfg["train"].get("warmup_steps", 2000)
    cfg["train"]["max_grad_norm"] = cfg["train"].get("max_grad_norm", 1.0)
    cfg["train"]["log_every"] = cfg["train"].get("log_every", 20)
    cfg["train"]["save_every"] = cfg["train"].get("save_every", 5000)
    cfg["train"]["start_step"] = int(cfg["train"].get("start_step", 0))
    cfg["train"]["init_encoder_from_pretrained"] = cfg["train"].get("init_encoder_from_pretrained", True)
    cfg["train"]["freeze_encoder"] = cfg["train"].get("freeze_encoder", True)
    cfg["train"]["use_teacher_target"] = cfg["train"].get("use_teacher_target", False)
    cfg["train"]["reconstruction_loss_weight"] = cfg["train"].get("reconstruction_loss_weight", 1.0)
    cfg["train"]["cosine_loss_weight"] = cfg["train"].get("cosine_loss_weight", 0.1)
    cfg["train"]["commitment_loss_weight"] = cfg["train"].get("commitment_loss_weight", 0.25)
    cfg["train"]["codebook_loss_weight"] = cfg["train"].get("codebook_loss_weight", 1.0)

    return cfg


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if manifest_path.suffix in {".txt", ".lst"}:
        return [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if manifest_path.suffix == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [item if isinstance(item, str) else item["image"] for item in payload]
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
    if pq is None:
        raise ImportError("pyarrow is required for parquet training.")
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


def build_transform(cfg):
    crop_size = cfg["model"]["image_size"]
    resize_size = cfg["data"].get("resize_shorter_edge", cfg["model"].get("resize_shorter_edge", crop_size))
    steps = [transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)]
    steps.append(transforms.RandomCrop(crop_size) if cfg["data"]["random_crop"] else transforms.CenterCrop(crop_size))
    if cfg["data"]["random_flip"]:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


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

        self.transform = build_transform(cfg)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


class ParquetImageDataset(IterableDataset):
    def __init__(self, cfg):
        parquet_root = cfg["data"].get("parquet_root")
        if not parquet_root:
            raise ValueError("data.parquet_root must be provided for parquet training.")
        self.paths = scan_parquet_files(parquet_root, recursive=cfg["data"]["recursive"])
        if not self.paths:
            raise ValueError("No parquet files were found.")
        self.transform = build_transform(cfg)

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
                        yield self.transform(decode_parquet_image(row["image"]))
                    except Exception:
                        continue


def build_dataset(cfg):
    if cfg["data"].get("parquet_root"):
        return ParquetImageDataset(cfg)
    return ImageOnlyDataset(cfg)


def build_model_args(cfg):
    return {
        "bottleneck": cfg["model"]["bottleneck"],
        "bottleneck_token_num": cfg["model"]["bottleneck_token_num"],
        "input_size": cfg["model"]["input_size"],
        "teacher": cfg["model"]["teacher"],
        "pool_scale": 1,
        "decoder_depth": cfg["model"]["decoder_depth"],
        "select_layer_id": cfg["model"]["select_layer_id"],
    }


def build_model(cfg):
    if cfg["train"].get("init_from"):
        model = TextAlignedTokenizer.from_checkpoint(
            cfg["train"]["init_from"],
            load_teacher=True,
            input_type="rec",
        )
    else:
        model = TextAlignedTokenizer(**build_model_args(cfg), input_type="rec")
        if cfg["train"]["init_encoder_from_pretrained"]:
            pretrained = AutoModel.from_pretrained(cfg["model"]["teacher"]).vision_model
            model.encoder.load_state_dict(pretrained.state_dict(), strict=False)
            del pretrained

    if cfg["train"]["freeze_encoder"]:
        model.encoder.requires_grad_(False)

    return model


def build_teacher_model(cfg):
    if not cfg["train"]["use_teacher_target"]:
        return None
    teacher = AutoModel.from_pretrained(cfg["model"]["teacher"]).vision_model
    teacher.config.output_hidden_states = True
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def select_hidden(outputs, select_layer_id):
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None and len(hidden_states) > 0:
        return hidden_states[select_layer_id]
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("Vision encoder did not return hidden_states or last_hidden_state.")
    return hidden


def preprocess_images(model, images):
    x = model.scale_layer(images)
    if tuple(x.shape[-2:]) != (model.input_size, model.input_size):
        x = model.image_resize(x)
    return x


def get_quantized_embedding(outputs):
    emb = outputs["emb"]
    indices = outputs["bottleneck_rep"]
    return F.embedding(indices.reshape(-1), emb).reshape(*indices.shape, emb.shape[-1])


def forward_train(model, images, pool_scale=1, teacher_model=None, select_layer_id=-2):
    x = preprocess_images(model, images)
    encoder_outputs = model.encoder(x, output_hidden_states=True, return_dict=True)
    student_hidden = select_hidden(encoder_outputs, select_layer_id)

    if teacher_model is not None:
        with torch.no_grad():
            teacher_outputs = teacher_model(x, output_hidden_states=True, return_dict=True)
            target_hidden = select_hidden(teacher_outputs, select_layer_id)
    else:
        target_hidden = student_hidden.detach()

    if pool_scale != 1:
        student_hidden = model.avg_pool(student_hidden, pool_scale)
        target_hidden = model.avg_pool(target_hidden, pool_scale)

    vq_feats = model.encode_task_layer(student_hidden.to(x))
    bottleneck_outputs = model.bottleneck(vq_feats)
    z = bottleneck_outputs.pop("output")
    pred_feats = model.decode(z)

    return {
        "pred_feats": pred_feats,
        "target_feats": target_hidden.detach(),
        "projected_z": bottleneck_outputs["projected_z"],
        "quantized_z": get_quantized_embedding(bottleneck_outputs),
        "bottleneck_rep": bottleneck_outputs["bottleneck_rep"],
        "emb": bottleneck_outputs["emb"],
    }


def compute_loss(outputs, cfg):
    pred = outputs["pred_feats"].float()
    target = outputs["target_feats"].float()
    projected_z = outputs["projected_z"].float()
    quantized_z = outputs["quantized_z"].float()

    recon_loss = F.mse_loss(pred, target)
    cosine_loss = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    commitment_loss = F.mse_loss(projected_z, quantized_z.detach())
    codebook_loss = F.mse_loss(quantized_z, projected_z.detach())

    loss = (
        cfg["train"]["reconstruction_loss_weight"] * recon_loss
        + cfg["train"]["cosine_loss_weight"] * cosine_loss
        + cfg["train"]["commitment_loss_weight"] * commitment_loss
        + cfg["train"]["codebook_loss_weight"] * codebook_loss
    )
    return loss, {
        "recon_loss": recon_loss.detach(),
        "cosine_loss": cosine_loss.detach(),
        "commitment_loss": commitment_loss.detach(),
        "codebook_loss": codebook_loss.detach(),
    }


def build_checkpoint(model, cfg, state_dict):
    return {
        "model": {
            "args": build_model_args(cfg),
            "sd": state_dict,
        },
        "meta": {
            "input_size": cfg["model"]["input_size"],
            "teacher": cfg["model"]["teacher"],
            "select_layer_id": cfg["model"]["select_layer_id"],
        },
    }


def save_checkpoint(accelerator, model, cfg, step):
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = accelerator.get_state_dict(model)
    checkpoint = build_checkpoint(accelerator.unwrap_model(model), cfg, state_dict)
    accelerator.save(checkpoint, output_dir / f"checkpoint-{step:07d}.pth")
    if accelerator.is_main_process:
        accelerator.save(checkpoint, output_dir / "latest.pth")
        accelerator.save(checkpoint, output_dir / "ta_tok.pth")


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


def main():
    args = parse_args()
    cfg = load_config(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["train"]["grad_accum_steps"],
        mixed_precision=cfg["train"]["mixed_precision"],
    )
    set_seed(cfg["train"]["seed"])

    if accelerator.is_main_process:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        with open(Path(cfg["output_dir"]) / "train_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    use_wandb = (
        accelerator.is_main_process
        and wandb is not None
        and os.environ.get("WANDB_MODE", "").lower() != "disabled"
    )
    if use_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "umllm-ta-tok"),
            name=os.environ.get("WANDB_RUN_NAME", Path(cfg["output_dir"]).name),
            config=cfg,
            dir=cfg["output_dir"],
        )

    dataset = build_dataset(cfg)
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
    teacher_model = build_teacher_model(cfg)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(
        trainable_params,
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

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    trainable_params_for_clip = [param for param in model.parameters() if param.requires_grad]
    if teacher_model is not None:
        teacher_model = teacher_model.to(accelerator.device)

    global_step = start_step
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "cosine_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "codebook_usage": 0.0,
    }
    step_count = 0

    while global_step < cfg["train"]["max_steps"]:
        for images in dataloader:
            pool_scale = random.choices(
                cfg["model"]["pool_scales"],
                weights=cfg["model"]["pool_scale_weights"],
                k=1,
            )[0]

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = forward_train(
                        model,
                        images,
                        pool_scale=pool_scale,
                        teacher_model=teacher_model,
                        select_layer_id=cfg["model"]["select_layer_id"],
                    )
                    loss, loss_parts = compute_loss(outputs, cfg)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params_for_clip, cfg["train"]["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                usage = outputs["bottleneck_rep"].unique().numel() / max(outputs["emb"].shape[0], 1)
                running["loss"] += accelerator.gather(loss.detach()).mean().item()
                running["codebook_usage"] += float(usage)
                for key, value in loss_parts.items():
                    running[key] += accelerator.gather(value).mean().item()
                step_count += 1

            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process and global_step % cfg["train"]["log_every"] == 0:
                    metrics = {key: value / max(step_count, 1) for key, value in running.items()}
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"step={global_step} "
                        f"loss={metrics['loss']:.4f} "
                        f"recon={metrics['recon_loss']:.4f} "
                        f"cos={metrics['cosine_loss']:.4f} "
                        f"commit={metrics['commitment_loss']:.4f} "
                        f"codebook={metrics['codebook_loss']:.4f} "
                        f"usage={metrics['codebook_usage']:.6f} "
                        f"pool_scale={pool_scale} "
                        f"lr={lr:.6e}"
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "train/step": global_step,
                                "train/loss": metrics["loss"],
                                "train/recon_loss": metrics["recon_loss"],
                                "train/cosine_loss": metrics["cosine_loss"],
                                "train/commitment_loss": metrics["commitment_loss"],
                                "train/codebook_loss": metrics["codebook_loss"],
                                "train/codebook_usage": metrics["codebook_usage"],
                                "train/pool_scale": pool_scale,
                                "train/lr": lr,
                            },
                            step=global_step,
                        )
                    running = {key: 0.0 for key in running}
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
