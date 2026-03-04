from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
    WhisperModel,
)

AUDIO_TOKEN = "<|audio_token|>"


def check_response(response: str) -> dict[str, str] | None:
    """Extract a strict {"question": "...", "answer": "..."} JSON object from a model response."""
    if not isinstance(response, str):
        return None
    response = response.strip()
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or set(data.keys()) != {"question", "answer"}:
        return None
    q = data["question"].strip() if isinstance(data.get("question"), str) else ""
    a = data["answer"].strip() if isinstance(data.get("answer"), str) else ""
    if not q or not a:
        return None
    return {"question": q, "answer": a}


# --------------------------
# Audio utils
# --------------------------
def load_wav_mono_16k(path: str | Path) -> torch.Tensor:
    """Load audio to mono float32 tensor [N] at 16 kHz."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [N, C]
    wav = torch.from_numpy(wav).transpose(0, 1)  # [C, N]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)  # [N]
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


def wav_to_whisper_features(
    wav_16k: torch.Tensor, feature_extractor: WhisperFeatureExtractor
) -> torch.Tensor:
    """wav_16k: [N] float32 -> whisper log-mel features [80, T] (float32)."""
    audio = wav_16k.detach().cpu().numpy()
    feats = feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
    return feats


# --------------------------
# Prompting utils
# --------------------------
def create_prompt(tokenizer, instruction: str, response: str | None, system_message: str = "") -> str:
    messages = []
    if system_message.strip():
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": instruction})

    add_generation_prompt = response is None
    if response is not None:
        messages.append({"role": "assistant", "content": response})

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


# --------------------------
# Model
# --------------------------
class AudioAdapter(nn.Module):
    # (B, T, D_w) -> (B, T_sub, D_llm)
    def __init__(self, whisper_dim: int, llm_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float, subsample_factor: int):
        super().__init__()
        self.k = subsample_factor
        in_dim = whisper_dim * self.k if self.k > 1 else whisper_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(max(num_layers - 1, 0))
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k > 1:
            B, T, D = x.shape
            T_trim = (T // self.k) * self.k
            x = x[:, :T_trim, :].reshape(B, T_trim // self.k, D * self.k)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.output_proj(x)


@dataclass
class ModelConfig:
    whisper_model: str = "openai/whisper-small"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_hidden_dim: int = 1024
    adapter_num_layers: int = 2
    adapter_dropout: float = 0.1
    subsample_factor: int = 4
    freeze_whisper: bool = True
    freeze_llm: bool = True


class AudioLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        whisper = WhisperModel.from_pretrained(cfg.whisper_model)
        self.whisper_encoder = whisper.encoder
        whisper_dim = self.whisper_encoder.config.d_model

        if cfg.freeze_whisper:
            for p in self.whisper_encoder.parameters():
                p.requires_grad = False

        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm_model, torch_dtype="auto", device_map=None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model, use_fast=True)

        self.tokenizer.add_special_tokens({"additional_special_tokens": [AUDIO_TOKEN]})
        self.audio_token_id = self.tokenizer(AUDIO_TOKEN, add_special_tokens=False).input_ids[0]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm.resize_token_embeddings(len(self.tokenizer))
        llm_dim = self.llm.config.hidden_size

        self.audio_adapter = AudioAdapter(
            whisper_dim=whisper_dim,
            llm_dim=llm_dim,
            hidden_dim=cfg.adapter_hidden_dim,
            num_layers=cfg.adapter_num_layers,
            dropout=cfg.adapter_dropout,
            subsample_factor=cfg.subsample_factor,
        )

        if cfg.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        # Needed by HF Trainer
        self.config = self.llm.config

    def encode_audio(self, input_features: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.cfg.freeze_whisper else torch.enable_grad()
        with ctx:
            w = self.whisper_encoder(input_features).last_hidden_state  # [B, T_w, D_w]
        return self.audio_adapter(w)  # [B, A, D_llm]

    @staticmethod
    def insert_audio_embeds(
        audio_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
        audio_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, A, D = audio_embeds.shape
        out_embeds = []
        out_labels = [] if labels is not None else None

        for i in range(B):
            pos = (input_ids[i] == audio_token_id).nonzero(as_tuple=False).flatten()
            if pos.numel() != 1:
                raise ValueError(f"Expected exactly 1 audio token, got {pos.numel()}")
            p = int(pos.item())
            merged = torch.cat([text_embeds[i, :p], audio_embeds[i], text_embeds[i, p+1:]], dim=0)
            out_embeds.append(merged)

            if labels is not None:
                lab = torch.cat([
                    labels[i, :p],
                    torch.full((A,), -100, device=labels.device, dtype=labels.dtype),
                    labels[i, p+1:]
                ], dim=0)
                out_labels.append(lab)

        out_embeds = torch.stack(out_embeds, dim=0)
        if labels is not None:
            out_labels = torch.stack(out_labels, dim=0)
        return out_embeds, out_labels

    def forward(self, audio_values: torch.Tensor, input_ids: torch.Tensor, attention_mask=None, labels=None):
        audio_embeds = self.encode_audio(audio_values)  # [B, A, D]
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, D]
        
        llm_dtype = text_embeds.dtype
        audio_embeds = audio_embeds.to(dtype=llm_dtype)

        merged_embeds, merged_labels = self.insert_audio_embeds(
            audio_embeds, text_embeds, input_ids, labels, self.audio_token_id
        )

        merged_attn = None
        if attention_mask is not None:
            B = input_ids.size(0)
            A = audio_embeds.size(1)
            merged = []
            for i in range(B):
                p = int((input_ids[i] == self.audio_token_id).nonzero(as_tuple=False).item())
                merged.append(torch.cat([
                    attention_mask[i, :p],
                    torch.ones(A, device=attention_mask.device, dtype=attention_mask.dtype),
                    attention_mask[i, p+1:]
                ], dim=0))
            merged_attn = torch.stack(merged, dim=0)

        return self.llm(
            inputs_embeds=merged_embeds,
            attention_mask=merged_attn,
            labels=merged_labels,
            return_dict=True,
        )

    @torch.inference_mode()
    def generate(
        self,
        audio_features: torch.Tensor,  # [1, 80, T]
        instruction: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """One-sample generation for notebook demos."""
        prompt = create_prompt(
            self.tokenizer,
            instruction=instruction,
            response=None,
        )
        tok = self.tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = tok["input_ids"].to(audio_features.device)
        attention_mask = tok["attention_mask"].to(audio_features.device)

        audio_embeds = self.encode_audio(audio_features)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        llm_dtype = text_embeds.dtype
        audio_embeds = audio_embeds.to(dtype=llm_dtype)

        merged_embeds, _ = self.insert_audio_embeds(
            audio_embeds, text_embeds, input_ids, labels=None, audio_token_id=self.audio_token_id
        )

        # build merged attention mask
        B = input_ids.size(0)
        A = audio_embeds.size(1)
        merged_attn = []
        for i in range(B):
            p = int((input_ids[i] == self.audio_token_id).nonzero(as_tuple=False).item())
            merged_attn.append(torch.cat([
                attention_mask[i, :p],
                torch.ones(A, device=attention_mask.device, dtype=attention_mask.dtype),
                attention_mask[i, p+1:]
            ], dim=0))
        merged_attn = torch.stack(merged_attn, dim=0)

        out = self.llm.generate(
            inputs_embeds=merged_embeds,
            attention_mask=merged_attn,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        gen_ids = out[0]
        txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return txt


# --------------------------
# Dataset + collator
# --------------------------
class AudioInstructDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_path: str | Path,
        tokenizer,
        feature_extractor: WhisperFeatureExtractor,
        max_length: int = 512,
        generation_mode: bool = False,
        num_samples: int | None = None,
    ):
        self.data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if num_samples is not None:
            self.data = self.data[:num_samples]
            print(self.data[0])
        self.tokenizer = tokenizer
        self.fe = feature_extractor
        self.max_length = max_length
        self.generation_mode = generation_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        wav = load_wav_mono_16k(s["audio_path"])
        feats = wav_to_whisper_features(wav, self.fe)  # [80, T]

        instruction = f"{AUDIO_TOKEN} Based on the given audio, answer: {s['question']}"
        answer = s["answer"]

        full_text = create_prompt(self.tokenizer, instruction, answer)
        tok = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding=False, return_tensors="pt")
        input_ids = tok["input_ids"][0]
        attention_mask = tok["attention_mask"][0]

        labels = None
        if not self.generation_mode:
            labels = input_ids.clone()
            prefix_text = create_prompt(self.tokenizer, instruction, response=None)
            prefix_ids = self.tokenizer(
                prefix_text, truncation=True, max_length=self.max_length, padding=False, return_tensors="pt"
            )["input_ids"][0]
            labels[:prefix_ids.numel()] = -100

        return {
            "audio_values": feats,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class AudioTextCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]):
        audio = torch.stack([b["audio_values"] for b in batch], dim=0)  # [B, 80, T]

        to_pad = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch]
        padded = self.tokenizer.pad(to_pad, padding=True, return_tensors="pt")

        max_len = padded["input_ids"].shape[1]
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            if b["labels"] is None:
                continue
            L = min(b["labels"].numel(), max_len)
            labels[i, :L] = b["labels"][:L]

        padded["labels"] = labels
        padded["audio_values"] = audio
        return padded


# --------------------------
# (De)serialization helpers
# --------------------------
def save_adapter_checkpoint(save_dir: str | Path, model: AudioLLM, cfg: ModelConfig):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": asdict(cfg),
            "audio_adapter": model.audio_adapter.state_dict(),
        },
        save_dir / "adapter.pt",
    )
    model.tokenizer.save_pretrained(save_dir / "tokenizer")
    # NOTE: whisper/llm weights are NOT saved here (usually huge). Expect to load them from HF cache.
    (save_dir / "README.txt").write_text(
        "This folder contains ONLY the trained audio adapter + tokenizer with the added AUDIO token.\n"
        "Whisper encoder / LLM are expected to be loaded from pretrained checkpoints.\n",
        encoding="utf-8",
    )


def load_adapter_checkpoint(load_dir: str | Path, device: torch.device | str = "cpu") -> tuple[AudioLLM, WhisperFeatureExtractor]:
    load_dir = Path(load_dir)
    ckpt = torch.load(load_dir / "adapter.pt", map_location="cpu")
    cfg = ModelConfig(**ckpt["model_cfg"])
    model = AudioLLM(cfg)
    # load tokenizer that includes special token
    model.tokenizer = AutoTokenizer.from_pretrained(load_dir / "tokenizer", use_fast=True)
    # re-register audio token id
    model.audio_token_id = model.tokenizer(AUDIO_TOKEN, add_special_tokens=False).input_ids[0]
    model.llm.resize_token_embeddings(len(model.tokenizer))
    model.audio_adapter.load_state_dict(ckpt["audio_adapter"], strict=True)
    model.to(device)
    fe = WhisperFeatureExtractor.from_pretrained(cfg.whisper_model)
    return model, fe
