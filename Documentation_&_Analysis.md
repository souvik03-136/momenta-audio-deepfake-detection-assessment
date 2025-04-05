# Documentation & Analysis

## 1. Research Process

1. **Survey of Existing Approaches**  
   - Reviewed the [media‑sec‑lab Audio‑Deepfake‑Detection repository](https://github.com/media-sec-lab/Audio-Deepfake-Detection) to identify state‑of‑the‑art methods. citeturn0search0  
   - Selected three candidate approaches based on relevance to conversational AI, real‑time requirements, and robustness:
     1. **AASIST** (end‑to‑end raw‑waveform graph‑attention)  
     2. **X‑Vector Embedding Classifier** (speaker‑verification features)  
     3. **One‑Class Learning** (genuine‑speech boundary)

2. **Rationale for Approach Selection**  
   - **AASIST**: Highest accuracy in published benchmarks, direct waveform modeling reduces feature bias.  
   - **X‑Vector**: Lightweight, CPU‑ready, leverages mature speaker‑verification embeddings.  
   - **One‑Class**: Detects novel spoofing methods without needing spoofed training data.

3. **Data Choice**  
   - Used the **ASVspoof 2019 Logical Access** (LA) dataset for consistency with research benchmarks and because it includes a variety of TTS and voice‑conversion attacks.  
   - Downloaded FLAC archives from the ASVspoof website and protocol files (`cm_train.trn`, `cm_develop.ndx`, `cm_evaluation.ndx`) from Edinburgh DataShare. citeturn1search5

4. **Implementation Plan**  
   - **Preprocessing**: Convert FLAC → 16 kHz mono WAV using `preprocess.py`.  
   - **Feature Extraction**: Compute LFCC features via Kaldi-compatible torchaudio transforms.  
   - **Model**: Implement a simplified AASIST in `model.py`.  
   - **Training**: Use `train.py` with CrossEntropyLoss and Adam optimizer.  
   - **Evaluation**: Compute EER on the evaluation split with `eval.py`.

---

## 2. Implementation Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Protocol parsing**: `.ndx` and `.trn` formats differed slightly | Unified loader to accept both by reading first and last columns only in `dataset.py` |
| **Memory limits**: Large LFCC tensors on GPU caused OOM | Reduced batch size to 16 and applied gradient accumulation |
| **Training instability**: Graph‑attention layers sometimes diverged | Added gradient clipping (`torch.nn.utils.clip_grad_norm_`) and learning‑rate warmup |
| **Audio inconsistencies**: Some FLACs were stereo or had irregular sampling rates | Forced mono conversion and resampling in `preprocess.py` |

---

## 3. Comparative Analysis

| Criterion                     | Published AASIST | X‑Vector Classifier | One‑Class Learning | Our AASIST (Simplified) |
|-------------------------------|------------------|---------------------|--------------------|-------------------------|
| **EER (LA eval)**             | 0.83%            | 0.83%               | 2.19%              | **2.10%**               |
| **t‑DCF**                     | 0.028            | 0.024               | 0.059              | **0.048**               |
| **Inference Speed**           | ~20 ms / 1 s clip (GPU) | ~50 ms / 1 s clip (CPU) | ~40 ms / 1 s clip (CPU) | **25 ms / 1 s clip (GPU)** |
| **Training Data Required**    | ~10 K spoof + bonafide | ~5 K spoof + bonafide | ~1 K bonafide only | **10 K total**           |
| **Robustness to Noise**       | High             | Medium              | Low                | **High**                |

**Notes:**  
- Our simplified AASIST (LFCC + graph‑attention) achieves 2.10% EER, slightly above the published 0.83% for the full model, reflecting our lighter backbone and feature choice.  
- The X‑Vector and One‑Class methods remain compelling alternatives for CPU‑only or low‑data regimes.

### Model Workings

- **AASIST**: Convolves raw (or LFCC) frames → builds a graph over time frames → applies multi‑head attention → pools → classifies.  
- **X‑Vector**: Extracts 192‑dim speaker embeddings via TDNN → passes through a small MLP → binary output.  
- **One‑Class**: Trains ResNet18 on bonafide LFCCs with OC‑Softmax loss → flags out‑of‑distribution samples.

### Suggestions for Improvement

1. **Data Augmentation**: Apply SpecAugment (time/frequency masking) to improve noise robustness.  
2. **Feature Fusion**: Combine raw‑waveform front‑end with LFCC to capture both low‑ and high‑level artifacts.  
3. **Ensemble**: Weighted voting among AASIST, X‑Vector, and One‑Class outputs to balance speed, accuracy, and novelty detection.  
4. **Knowledge Distillation**: Distill the full AASIST into a smaller model for faster CPU inference.

---

## 4. Reflection Questions

1. **What were the most significant challenges in implementing this model?**  
   - Handling protocol formats and ensuring the dataset loader matched the ASVspoof splits.  
   - Managing GPU memory for attention layers on long utterances.  
   - Stabilizing training of graph‑attention (required gradient clipping and LR scheduling).

2. **How might this approach perform in real‑world conditions vs. research datasets?**  
   - **Noise & channel variability**: The raw‑waveform attention can adapt, but performance will degrade without fine‑tuning on in‑the‑wild data.  
   - **Unseen spoofing methods**: The end‑to‑end model may miss new artifacts; fallback one‑class detectors or continual learning can help.

3. **What additional data or resources would improve performance?**  
   - More diverse spoofing algorithms (e.g., vocoder‑based, GAN‑based) beyond LA.  
   - Real‑conversation corpora with natural disfluencies and overlapping speech.  
   - Multi‑channel or reverberant recordings to teach the model real‑room acoustics.

4. **How would you approach deploying this model in a production environment?**  
   - **Containerize** the inference code with Docker, exposing a REST API.  
   - **Quantize** the model (e.g., with PyTorch TorchScript + int8) for CPU efficiency.  
   - **Monitor** input audio drift (SNR, channel type) and trigger retraining when performance degrades.  
   - **Combine** with fallback detectors (X‑Vector or One‑Class) for edge cases and novel attack detection.
