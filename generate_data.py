"""
Generate instruction-style QA data from LibriSpeech transcripts.

Output: audio_llm/artifacts/instruct_data.json
Each item:
{
  "audio_path": ".../xxx.flac",
  "transcription": "...",
  "question": "...",
  "answer": "..."
}

Run (from the audio_llm folder):
python generate_data.py \
  --librispeech_root ../LibriSpeech/dev-clean \
  --num_samples 2500 \
  --batch_size 32 \
  --gen_model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations 

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from audio_llm_lib import check_response


BASE_SYSTEM_MESSAGE = r"""
You are a helpful assistant.

TASK:
Given a transcript, generate exactly ONE question that can be answered from the transcript, and a short answer.

OUTPUT RULES (STRICT):
- Output MUST be a single valid JSON object and NOTHING else.
- No markdown, no backticks, no extra text, no explanations.
- Use double quotes for JSON strings.
- Only these keys are allowed: "question", "answer".
- Both values must be non-empty strings.

OUTPUT SCHEMA:
{"question": "<string>", "answer": "<string>"}

NOW FOLLOW THE OUTPUT RULES EXACTLY.
""".strip()


def load_librispeech_transcripts(root: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for trans_file in root.rglob("*.trans.txt"):
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                utt_id = parts[0]
                text = " ".join(parts[1:])
                transcripts[utt_id] = text
    return transcripts


class QADataGeneratorHF:
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        system_message: str = BASE_SYSTEM_MESSAGE,
        generation_kwargs: dict | None = None,
        local_dir: Path | None = None,
    ):
        self.model_path = model_path
        self.system_message = system_message
        self.device = device
        self.local_dir = local_dir

        generation_kwargs = {} if generation_kwargs is None else dict(generation_kwargs)
        self.generation_kwargs = generation_kwargs

        if local_dir is not None:
            local_dir.mkdir(parents=True, exist_ok=True)

        if local_dir is not None and any(local_dir.iterdir()):
            print(f"Loading model from local directory: {local_dir}")
            load_path = str(local_dir)
            local_files_only = True
        else:
            print(f"Downloading model from HF: {model_path}")
            load_path = model_path
            local_files_only = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path, use_fast=True, local_files_only=local_files_only
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=local_files_only,
        ).eval()

        self.generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        if not local_files_only and local_dir is not None:
            print(f"Saving model to {local_dir}")
            self.tokenizer.save_pretrained(local_dir)
            self.model.save_pretrained(local_dir)

    @torch.inference_mode()
    def generate_batch(self, transcriptions: list[str]) -> list[str]:
        texts = []
        for t in transcriptions:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": t},
            ]
            txt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(txt)

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model.generate(**inputs, **self.generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        results = []
        for i in range(out.shape[0]):
            gen_ids = out[i, prompt_len:]
            results.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_root", type=str, required=True,
                        help="Path to LibriSpeech/dev-clean (folder that contains speaker subfolders)")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts",
                        help="Where to save instruct_data.json (relative to current working dir)")
    parser.add_argument("--num_samples", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gen_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--gen_model_local_dir", type=str, default="artifacts/gen_model")

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    root_dir = Path(args.librispeech_root)
    assert root_dir.exists(), f"Not found: {root_dir}"

    transcripts = load_librispeech_transcripts(root_dir)
    print("Loaded transcripts:", len(transcripts))

    librispeech_data = []
    for flac_path in sorted(root_dir.rglob("*.flac")):
        utt_id = flac_path.stem
        text = transcripts.get(utt_id)
        if text is None:
            continue
        librispeech_data.append({"audio_path": str(flac_path), "transcription": text})
    print("Audio/transcript pairs:", len(librispeech_data))

    sample = librispeech_data[: args.num_samples]

    generator = QADataGeneratorHF(
        model_path=args.gen_model,
        device=device,
        local_dir=Path(args.gen_model_local_dir),
        generation_kwargs=dict(
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
        ),
    )

    good, bad = [], []
    for start in tqdm(range(0, len(sample), args.batch_size), desc="QA generation (batched)"):
        batch = sample[start : start + args.batch_size]
        transcriptions = [s["transcription"] for s in batch]
        raw_list = generator.generate_batch(transcriptions)
        for s, raw in zip(batch, raw_list):
            qa = check_response(raw)
            if qa is None:
                bad.append({**s, "raw_response": raw})
            else:
                good.append({**s, **qa})

    print("Total:", len(sample))
    print("Good :", len(good))
    print("Bad  :", len(bad))

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    out_path = artifacts_dir / "instruct_data_train.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(good, f, ensure_ascii=False, indent=2)
    print("Saved:", out_path.resolve())

    if bad:
        bad_path = artifacts_dir / "instruct_data_bad_train.json"
        with open(bad_path, "w", encoding="utf-8") as f:
            json.dump(bad[:200], f, ensure_ascii=False, indent=2)
        print("Also saved first 200 bad samples:", bad_path.resolve())


if __name__ == "__main__":
    main()
