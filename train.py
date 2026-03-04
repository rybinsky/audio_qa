from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Subset
from transformers import TrainingArguments, Trainer, TrainerCallback, WhisperFeatureExtractor, EarlyStoppingCallback

from audio_llm_lib import (
    AudioInstructDataset,
    AudioLLM,
    AudioTextCollator,
    ModelConfig,
    save_adapter_checkpoint,
)


@dataclass
class ExperimentConfig:
    # ---- run ----
    artifacts_dir: str = "artifacts"
    run_name: Optional[str] = "early_stopping_full"
    seed: int = 42
    tb_root: str = "tb_logs"

    # ---- data ----
    data_json: str = "artifacts/instruct_data_train.json"
    num_samples: Optional[int] = None
    val_pct: float = 0.03
    max_length: int = 512

    # ---- model ----
    whisper_model: str = "openai/whisper-small"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    subsample_factor: int = 4
    adapter_hidden_dim: int = 1024
    adapter_num_layers: int = 2
    adapter_dropout: float = 0.1

    freeze_whisper: bool = True   # adapter-only by default
    freeze_llm: bool = True       # adapter-only by default

    # ---- train ----
    steps: int = 10000
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 3e-4
    logging_steps: int = 500
    eval_steps: int = 500
    use_early_stopping: bool = True

    warmup_steps: int = 0
    weight_decay: float = 0.0
    
    # ---- early stopping ----
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0

    # ---- checkpointing ----
    save_steps: int = 500
    save_total_limit: int = 2

    def resolved_run_name(self) -> str:
        return self.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_dir(self) -> Path:
        return Path(self.artifacts_dir) / "runs" / self.resolved_run_name()

    def tb_logdir(self) -> Path:
        return Path(self.tb_root) / self.resolved_run_name()

    def save(self) -> None:
        run_dir = self.run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_val_split(dataset, val_pct: float, seed: int = 42):
    if not (0.0 < val_pct < 1.0):
        raise ValueError("val_pct должен быть в (0, 1), например 0.1")
    n = len(dataset)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_val = int(round(n * val_pct))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())


class LossHistory(TrainerCallback):
    def __init__(self):
        self.steps, self.losses = [], []
        self.eval_steps, self.eval_losses = [], []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            self.steps.append(state.global_step)
            self.losses.append(float(logs["loss"]))
        if "eval_loss" in logs:
            self.eval_steps.append(state.global_step)
            self.eval_losses.append(float(logs["eval_loss"]))


def find_latest_trainer_checkpoint(trainer_out_dir: Path) -> Optional[str]:
    """
    Ищем последний checkpoint-* внутри trainer_out_dir.
    Возвращаем путь строкой (как ожидает HF Trainer) или None.
    """
    if not trainer_out_dir.exists():
        return None
    ckpts = [p for p in trainer_out_dir.glob("checkpoint-*") if p.is_dir()]
    if not ckpts:
        return None

    def step_num(p: Path) -> int:
        # checkpoint-1234 -> 1234, иначе -1
        try:
            return int(p.name.split("-")[-1])
        except Exception:
            return -1

    ckpts.sort(key=step_num)
    latest = ckpts[-1]
    return str(latest)


def main():
    cfg0 = ExperimentConfig()

    set_seed(cfg0.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print("DEVICE:", device, "DTYPE:", dtype)

    artifacts_dir = Path(cfg0.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data_json = Path(cfg0.data_json)
    if not data_json.exists():
        raise FileNotFoundError(f"Not found: {data_json} (run generate_data.py first)")

    run_dir = cfg0.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    tb_logdir = cfg0.tb_logdir()
    tb_logdir.mkdir(parents=True, exist_ok=True)
    print(f"Tensorboard logdir: '{tb_logdir}'")
    
    cfg0.save()

    cfg = ModelConfig(
        whisper_model=cfg0.whisper_model,
        llm_model=cfg0.llm_model,
        adapter_hidden_dim=cfg0.adapter_hidden_dim,
        adapter_num_layers=cfg0.adapter_num_layers,
        adapter_dropout=cfg0.adapter_dropout,
        subsample_factor=cfg0.subsample_factor,
        freeze_whisper=cfg0.freeze_whisper,
        freeze_llm=cfg0.freeze_llm,
    )

    model = AudioLLM(cfg).to(device)
    if device.type == "cuda":
        model.whisper_encoder.to(device, dtype=dtype)
        model.audio_adapter.to(device, dtype=dtype)
        model.llm.to(device)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.whisper_model)
    collator = AudioTextCollator(model.tokenizer)

    dataset = AudioInstructDataset(
        json_path=data_json,
        tokenizer=model.tokenizer,
        feature_extractor=feature_extractor,
        max_length=cfg0.max_length,
        num_samples=cfg0.num_samples,
    )
    train_dataset, eval_dataset = train_val_split(dataset, val_pct=cfg0.val_pct, seed=cfg0.seed)
    print(f"Train set size: {len(train_dataset)}, eval set size: {len(eval_dataset)}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    loss_cb = LossHistory()

    trainer_out_dir = run_dir / "trainer_out"
    training_args = TrainingArguments(
        output_dir=str(trainer_out_dir),
        per_device_train_batch_size=cfg0.batch_size,
        gradient_accumulation_steps=cfg0.grad_accum,
        max_steps=cfg0.steps,
        learning_rate=cfg0.lr,
        warmup_steps=cfg0.warmup_steps,
        weight_decay=cfg0.weight_decay,
        logging_steps=cfg0.logging_steps,
        evaluation_strategy="steps",
        eval_steps=cfg0.eval_steps,

        save_strategy="steps",
        save_safetensors=False,
        save_steps=cfg0.save_steps,
        save_total_limit=cfg0.save_total_limit,
        
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to=["tensorboard"],
        logging_dir=str(tb_logdir),
        remove_unused_columns=False,
        bf16=(device.type == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=
            [loss_cb] + (
                [
                    EarlyStoppingCallback(
                        early_stopping_patience=cfg0.early_stopping_patience,
                        early_stopping_threshold=cfg0.early_stopping_threshold,
                    )
                ] if cfg0.use_early_stopping else []
            )
    )

    resume_ckpt = find_latest_trainer_checkpoint(trainer_out_dir)
    if resume_ckpt is not None:
        print("Resuming from checkpoint:", resume_ckpt)
    else:
        print("No checkpoint found, training from scratch.")

    train_out = trainer.train(resume_from_checkpoint=resume_ckpt)
    print(train_out)

    metrics = trainer.evaluate()
    print(metrics)

    save_adapter_checkpoint(run_dir, model, cfg)
    print("Saved adapter checkpoint to:", run_dir.resolve())
    print("TensorBoard logdir:", tb_logdir.resolve())


if __name__ == "__main__":
    main()
