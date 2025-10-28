# tone-toxicity

# Measuring Multimodal Effects on Toxicity Judgments in Commentary Media

## Project Summary
This project develops a multimodal deep learning network that quantifies toxicity in social commentary clips and tests whether video and audio signals change the model’s toxicity prediction relative to text only. We will train and evaluate on segments from **Ben Shapiro**, **Joe Rogan**, and **Jon Stewart**, and predict a single continuous toxicity intensity score ranging from **0 to 1**.

Labels are created using LLM-assisted generation of weak/noisy labels. Each segment receives an output containing the predicted toxicity score. These weak labels will then be refined manually to create a smaller, high-quality evaluation set for our true labels. The model uses a shared transformer encoder (**RoBERTa** or **DeBERTa**) for text representation. First, we build a text-only baseline model, and then add audio and video features.

### Models Compared
- **Model A (Text-only)**
- **Model B (Text+Audio)**

### Evaluation Focus
We will measure how the predicted toxicity score changes as additional features are introduced. Metrics include:
- **Mean Absolute Error (MAE)** against true labels  
- **Mean absolute score shift** between models  
- **Correlation** of toxicity scores across modalities

---

## What We Will Do

### Step 1 – Data Collection & Preprocessing
Scrape short clips (10–45 seconds) from publicly available YouTube segments of Ben Shapiro, Joe Rogan, and Jon Stewart. Transcripts can be extracted using the YouTube API or Whisper.

### Step 2 – Label Creation (Weak vs. Curated)
Use an LLM (Phi-3-mini / Llama-3-Instruct / fine-tuned RoBERTa / DeBERTa) to build one output per observation.

### Step 3 – Toxicity Scoring
Feed LLM output into an off-the-shelf toxicity scorer.  
**Output:** `{toxicity_score: [0, 1]}`.

### Step 4 – Evaluation & Analysis
Evaluation will focus on how the predicted toxicity score changes as additional features are introduced. Metrics include MAE against true labels, mean absolute score shift between models, and correlation across modalities.

**Audio branch:** Introduce features such as pitch, energy, and tempo, or combine into embeddings.  
**Combined/fused model:** Training will begin with Model A (text-only), with the addition of audio features for Model B while maintaining fixed hyperparameters to isolate effects.

**Train and compare the two models:**
- Model A (Text-only)  
- Model B (Text+Audio)

---

## Resources and Related Work

### Models/APIs to Score Toxicity in Text
- Perspective API: <https://developers.perspectiveapi.com/s/?language=en_US>  
- Detoxify: <https://github.com/unitaryai/detoxify>

### Transformer Models for Text
- RoBERTa  
- DeBERTa  
- T5

### Models for Transcription
- OpenAI Whisper: <https://github.com/openai/whisper>

### Models for Video (If Needed)
- VideoMAE: <https://arxiv.org/abs/2203.12602>  
- TimeSformer: <https://arxiv.org/abs/2102.05095>

---

## Datasets
- **Primary Source:** Public YouTube clips from Ben Shapiro, Joe Rogan, and Jon Stewart.  
- **Transcripts and Audio:** Extracted or cleaned with Whisper/WhisperX.  
- **Weak Labels:** LLM-generated JSON outputs containing toxicity scores.  
- **True Labels:** Human refined.

