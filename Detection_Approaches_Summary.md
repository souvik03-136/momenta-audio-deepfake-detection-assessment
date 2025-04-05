**Audio Deepfake Detection Approaches for Conversational AI**  
*Analysis of Three Promising Models for Real‑Time Detection in Natural Dialogues*

---

## 1. AASIST (Raw‑Waveform Graph‑Attention)  
**Description**  
AASIST ingests raw audio (no handcrafted features) and constructs a spectro‑temporal graph over short frames. A heterogeneous graph‑attention layer then highlights suspicious artifacts across both frequency and time domains. citeturn1search0  

**Key Metrics**  
- Equal Error Rate (EER): 0.83%  
- t‑DCF: 0.028  

**Strengths**  
- **Low latency**: Processes one‑second segments in under 25 ms on a modern GPU.  
- **Robustness**: Learns directly from waveforms, making it resilient to background noise and variable recording conditions.  

**Trade‑offs**  
- Requires on the order of 10 000 labeled examples to achieve top performance.  
- Graph‑attention mechanisms can be difficult to interpret in forensic settings.  

---

## 2. X‑Vector Speaker‑Embedding Classifier  
**Description**  
Uses pre‑trained TDNN x‑vector embeddings (originally for speaker verification) as input to a lightweight classifier (e.g., SENet34). Deviations from a speaker’s typical embedding distribution signal a potential deepfake. citeturn2search8  

**Key Metrics**  
- EER: 0.83%  
- t‑DCF: 0.024  

**Strengths**  
- **CPU‑friendly**: Embedding extraction plus a small classifier runs in real time on standard hardware.  
- **Multilingual support**: Pre‑trained x‑vector models exist for over 100 languages.  

**Trade‑offs**  
- Vulnerable to high‑quality voice clones of the same speaker.  
- May misclassify benign channel artifacts (e.g., telephony codecs) as spoofing.  

---

## 3. One‑Class Learning on Genuine Speech  
**Description**  
Trains a one‑class model (e.g., ResNet18 with LFCC features) solely on bona‑fide recordings. Anything falling outside the learned “real speech” distribution is flagged as a spoof. citeturn2search8  

**Key Metrics**  
- EER: 2.19%  
- t‑DCF: 0.059  

**Strengths**  
- **Novel‑attack detection**: Can identify unseen spoofing methods without retraining.  
- **Data efficiency**: Achieves reasonable performance with as few as 1 000 real‑speech examples.  
- **Explainability**: Based on well‑understood spectral features (LFCCs).  

**Trade‑offs**  
- Higher false‑alarm rate (5–7%) on genuine speech, requiring downstream filtering.  
- Performance degrades on very short utterances (<3 s).  

---

## Comparative Overview

| Criterion                    | AASIST            | X‑Vector Embedding | One‑Class Learning |
|------------------------------|-------------------|--------------------|--------------------|
| **Inference Speed**          | GPU‑optimized     | CPU‑ready          | Moderate           |
| **Training Data Required**   | High (~10 K clips)| Medium (~5 K clips)| Low (~1 K clips)   |
| **Detection of Novel Attacks** | Good            | Fair               | Excellent          |
| **Robustness to Noise**      | High              | Medium             | Low                |

---

## Recommendation for Momenta  
Implement a **hybrid pipeline** that leverages:  
1. **AASIST** as the primary detector in high‑risk or high‑quality audio streams.  
2. **X‑Vector** screening for low‑latency, resource‑constrained scenarios.  
3. **One‑Class** monitoring to catch emerging or unknown spoofing techniques.
