# README.md

## 📌 Audio QA (Audio Question Answering)

Проект по **Audio Question Answering**: модель принимает аудиосигнал и текстовый вопрос и генерирует текстовый ответ.

Архитектура построена на основе:

* 🎙️ энкодера **OpenAI** **Whisper**
* 💬 LLM **Alibaba** **Qwen2.5-1.5B-Instruct**
* 🔗 обучаемого audio-adapter, который проецирует Whisper-эмбеддинги в пространство LLM

Обучение производится в режиме **adapter-only** (Whisper и LLM заморожены по умолчанию).

---

# 🧠 Архитектура

```
Audio (.wav / .flac)
        ↓
Whisper encoder (log-mel → hidden states)
        ↓
AudioAdapter (subsampling + MLP blocks)
        ↓
LLM embeddings space
        ↓
Qwen (causal LM)
        ↓
Text answer
```

### Ключевая идея

В prompt вставляется специальный токен:

```
<|audio_token|>
```

Во время forward-pass этот токен заменяется на последовательность аудио-эмбеддингов.

---

# 📂 Структура проекта

```
audio_qa/
│
├── audio_llm_lib.py        # модель, датасет, adapter, utils
├── generate_data.py        # генерация QA датасета из LibriSpeech
├── train.py                # обучение через HF Trainer
│
├── artifacts/
│   ├── instruct_data_train.json
│   └── runs/
│
├── tb_logs/                # TensorBoard логи
│
└── view_results.ipynb      # ноутбук для анализа результатов
```

---

# 📊 Датасет

Используется **LibriSpeech**.

Из транскриптов автоматически генерируются пары:

```json
{
  "audio_path": "...flac",
  "transcription": "...",
  "question": "...",
  "answer": "..."
}
```

---

# 🚀 Генерация QA датасета

```bash
python generate_data.py \
  --librispeech_root ../LibriSpeech/dev-clean \
  --num_samples 2500 \
  --batch_size 32
```

Результат:

```
artifacts/instruct_data_train.json
```

---

# 🏋️ Обучение

```bash
python train.py
```

По умолчанию:

* adapter-only training
* early stopping
* логирование в TensorBoard
* автоматический resume с checkpoint

После обучения сохраняется:

```
artifacts/runs/<run_name>/
    ├── adapter.pt
    ├── tokenizer/
    ├── config.json
    └── trainer_out/
```

---

# 📈 TensorBoard

```bash
tensorboard --logdir tb_logs
```

---

# 🧪 Инференс

После обучения можно загрузить adapter:

```python
from audio_llm_lib import load_adapter_checkpoint

model, feature_extractor = load_adapter_checkpoint("artifacts/runs/<run_name>")
```

---

Скажи, в каком формате тебе это нужно — курсовая, диплом, github-портфолио или research prototype?
