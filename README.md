# SimulStreaming (Extended)

SimulStreaming implements Whisper model for translation and transcription in
simultaneous mode (known as *streaming* in the ASR community).
It uses the state-of-the-art simultaneous policy **AlignAtt**, enabling fast and efficient decoding.

SimulStreaming merges:
- https://github.com/backspacetg/simul_whisper/
- https://github.com/ufal/whisper_streaming

We acknowledge and thank the original authors of these repositories for their foundational work.

---

## ✨ Extended Contribution

This fork introduces an additional utility developed by **Rodolfo Zevallos (Barcelona Supercomputing Center - BSC)**.

### 🔧 New Feature: Whisper Model Conversion (2-files → 1-file)

Whisper models from Hugging Face are typically stored as multiple files (fragmented format).  
However, the original Whisper implementation expects a **single `.pt` checkpoint file**.

This repository adds a method to:

- Convert Hugging Face Whisper checkpoints → OpenAI-compatible `.pt`
- Fix key mismatches between implementations
- Reconstruct internal architecture mappings
- Generate required metadata (`dims`)

This enables seamless usage of Hugging Face-trained Whisper models in streaming pipelines.

---

## ⚙️ Conversion Script

The script:

`join2bin.py`

### What it does (technical overview)

- Loads Hugging Face model via `WhisperForConditionalGeneration`
- Extracts `state_dict`
- Applies critical key remapping:
  - `decoder.embed_tokens.weight → decoder.token_embedding.weight`
  - `decoder.layer_norm → decoder.ln`
- Converts architecture naming:
  - `.layers → .blocks`
  - `k_proj/q_proj/v_proj → key/query/value`
- Fixes positional embeddings
- Removes incompatible keys (`proj_out.weight`)
- Builds OpenAI-compatible metadata (`dims`)
- Saves unified checkpoint

---

## 🚀 Usage of the Conversion Method

### 1. Configure paths

Edit inside `join2bin.py`:

```python
MODEL_PATH_FRAGMENTED = "./your-hf-model"
OUTPUT_PATH = "./converted_model"
OUTPUT_MODEL_FILENAME = "model.pt"
```

### 2. Run the script

```bash
python join2bin.py
```

### 3. Output

```
converted_model/model.pt
```

This file is directly compatible with SimulStreaming.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Optional (lighter install):
- Remove `torchaudio` if you do not need VAD (`--vac`)

---

## 🚀 Running SimulStreaming

### ▶️ 1. Basic streaming from audio file

```bash
python3 simulstreaming_whisper.py audio.wav \
  --model_path ./converted_model/model.pt \
  --language en \
  --task transcribe
```

---

### 🌍 2. Translation mode

```bash
python3 simulstreaming_whisper.py audio.wav \
  --model_path ./converted_model/model.pt \
  --language cs \
  --task translate
```

---

### ⚡ 3. Computationally unaware simulation (lower-bound latency)

```bash
python3 simulstreaming_whisper.py audio.wav \
  --model_path ./converted_model/model.pt \
  --language en \
  --task transcribe \
  --comp_unaware
```

---

### 🎯 4. Recommended settings (with VAD + beam search)

```bash
python3 simulstreaming_whisper.py audio.wav \
  --model_path ./converted_model/model.pt \
  --language en \
  --task transcribe \
  --vac \
  --beams 5 \
  --min-chunk-size 1.0
```

---

## 🎙️ Real-time Streaming from Microphone (Server)

Start server:

```bash
python3 simulstreaming_whisper_server.py \
  --model_path ./converted_model/model.pt \
  --language en \
  --task transcribe \
  --port 43001
```

Linux client example:

```bash
arecord -f S16_LE -c1 -r 16000 -t raw -D default | nc localhost 43001
```

Mac/Windows alternative:

```bash
ffmpeg -f avfoundation -i ":0" -ac 1 -ar 16000 -f s16le - | nc localhost 43001
```

---

## 🧠 Why this matters

This extension resolves a key incompatibility:

| Format | Issue |
|------|------|
| Hugging Face | Fragmented checkpoints |
| Whisper (OpenAI-style) | Requires single `.pt` file |

This method bridges both ecosystems, enabling:

- Reuse of fine-tuned Whisper models  
- Compatibility with streaming pipelines  
- Simplified deployment  

---

## 📊 Example Full Workflow

```bash
# Step 1: Convert HF model
python join2bin.py

# Step 2: Run streaming
python3 simulstreaming_whisper.py audio.wav \
  --model_path ./converted_model/model.pt \
  --language en \
  --task transcribe
```

---

## 📣 Acknowledgements

We sincerely thank:

- Simul-Whisper authors  
- Whisper-Streaming contributors  
- Original Whisper implementation  

This fork builds directly on their work.

---

