# OCR & Summarization Micro-service

**by Aleksei Vdonin**  
Faculty of Computer Science, HSE University, Moscow, 2025  

## Coursework 2025  

A self-hosted micro-service that **recognises and summarises HSE corporate documents** (PDF, DOC/DOCX, TXT) and is ready to be plugged into the mobile application **HSE App X**.  
The pipeline can run end-to-end on **CPU-only servers inside the HSE perimeter**, keeping sensitive documents private.

---

## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Dataset](#dataset)  
4. [Training & Distillation](#training--distillation)  
5. [Results](#results)  
6. [Web Demo](#web-demo)  
7. [Future Work](#future-work)  
8. [Links](#links)  
9. [License](#license)  

---

## Overview  

| Stage | Description |
|-------|-------------|
| **1. Crawling & OCR** | Parses >2 000 public documents from *hse.ru* and runs Tesseract OCR where needed. |
| **2. Dataset Builder** | Splits texts into 1 000-token chunks with 100-token overlap, then queries OpenAI o3-mini to create reference summaries. |
| **3. Teacher** | Fine-tunes `d0rj/ru-mbart-large-summ` to reach state-of-the-art quality on legal prose. |
| **4. Students** | Distils the teacher into compact BiLSTM and BiLSTM + MHA Seq2Seq models for CPU inference. |
| **5. Streamlit UI** | Lets users upload a file or paste text and instantly compare outputs of all four models. |

All components are container-friendly and require only **Python 3.11 + PyPI packages**.

---

## Features  

- **Document-scale abstractive summarization** (arbitrary length, sliding window).  
- **Soft-label knowledge distillation** for ~5× faster CPU inference.  
- **End-to-end automation**: call → OCR → chunk → target → summary.  
- **Lightweight deployment** (no GPU, < 200 MB student weights).  
- **Interactive web demo** with timing & progress bars.

---

## Dataset  

| Stage | Output | Size |
|-------|--------|------|
| Crawl portal | `data/data_raw/` | 2 059 docs |
| OCR & cleaning | `data/rproc_data/` | 2 059 UTF-8 texts |
| Chunking | 8 000 + fragments | ≈ 860 tokens each |
| Final JSONL | `train_smart.jsonl` | ~67 MB (text, summary pairs) |

---

## Training & Distillation  

| Stage | Entry-point | Epochs | GPU-hours |
|-------|-------------|--------|-----------|
| **Teacher fine-tune** | `train/mBART.ipynb` | 7 | 27 |
| **LSTM student** | `distill/distill_train.ipynb` | 50 | 30 |
| **LSTM + MHA student** | `distill/distill_train_mha.ipynb` | 16 | 20 |

Experiments are logged with **Weights & Biases**; reproducible configs are present in respective folders.

---

## Results  

| Model | Params | BERTScore F1 | CPU latency (500 w) | Speed-up |
|-------|--------|-------------:|--------------------:|---------:|
| **Teacher (mBART-large)** | 380 M | **0.76** | 11.2 s | 1× |
| Student (LSTM + MHA) | 47 M | 0.69 | 4.66 s | 2.4× |
| Student (LSTM) | 46 M | 0.68 | **2.23 s** | 5.0× |

---

## Web Demo  

The Streamlit interface supports:

1. **File upload** (TXT, DOC/DOCX, PDF with on-the-fly OCR).  
2. **Real-time comparison** of all models with elapsed-time badges.  
3. **Progress bars** for OCR & long-text chunking.

A live instance runs at **`http://158.160.61.64:8501/`** (internal HSE network).

---

## Future Work  

- **Hybrid extractive + abstractive** summarization for better factual coverage.  
- **Dynamic model selector** (fast vs. accurate) based on document length / SLA.  
- **ONNX export** of student models for mobile devices.  

---

## Links  

- **Live UI** – *[Link](http://158.160.61.64:8501/)*  
- **Train Data** – *[Drive](https://drive.google.com/drive/folders/1Q6w4pFyT_C4i-YpwldmFK5zU__c73QH8?usp=sharing)*  
- **Model Checkpoints** – *[Drive](https://drive.google.com/drive/folders/1dRNEiKeLWVwxjA9b1gviGRu0oAXL2mh0?usp=sharing)*
- **Coursework Report** - *[Report](https://drive.google.com/file/d/15s1vvJh0rRfqZ_qWpkAo9C-Rkl1l--Nh/view?usp=sharing)*

---

## License  

Released under the **MIT License** – see [`LICENSE`](LICENSE) for full text.
