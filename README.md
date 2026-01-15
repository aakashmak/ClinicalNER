# 🧬 ClinicalNER — Biomedical Named Entity Recognition with NVIDIA NeMo

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NeMo](https://img.shields.io/badge/NVIDIA-NeMo-76b900)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)

A production-oriented **Biomedical NER pipeline** built on **NVIDIA NeMo's pretrained BioBERT token classification model** to extract structured clinical entities from unstructured medical text — clinical trial abstracts, EHR notes, and CDISC-standard documentation.

---

## 🎯 Problem Statement

Pharmaceutical and clinical research organizations process thousands of unstructured clinical trial documents. Manually identifying drugs, dosages, adverse events, and trial phases is time-consuming and error-prone. ClinicalNER automates this using NVIDIA NeMo's domain-adaptive NLP stack.

---

## ✨ Features

- **NVIDIA NeMo BioBERT** — pretrained token classification model fine-tuned for biomedical NER
- **8 Clinical Entity Types** — Drugs, Diseases, Dosages, Adverse Events, Trial Phases, Biomarkers, Patient Populations, Procedures
- **Interactive Streamlit UI** — annotated text highlighting with color-coded entity badges
- **Export Results** — CSV and JSON download for downstream pipeline integration
- **3 Built-in Sample Abstracts** — Pfizer, Moderna, AbbVie trial data from ClinicalTrials.gov
- **Graceful Fallback** — rule-based tagger if NeMo is not available (CPU-friendly demo mode)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI (app.py)               │
│  Input Text → Annotated Output → CSV/JSON Export    │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│            ClinicalNEREngine (src/ner_engine.py)     │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │         NVIDIA NeMo NLP Stack               │    │
│  │  TokenClassificationModel (ner_en_bert)     │    │
│  │  BioBERT backbone → BIO tag sequence        │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  Fallback: Rule-based keyword + regex tagger         │
└─────────────────────────────────────────────────────┘
```

---

## 🧬 Entity Types Detected

| Entity | Color | Example |
|--------|-------|---------|
| `DRUG` | 🔴 Red | BNT162b2, venetoclax, azacitidine |
| `DISEASE` | 🩵 Teal | COVID-19, AML, neutropenia |
| `DOSAGE` | 🟡 Yellow | 30 mcg, 400 mg daily, 75 mg/m² |
| `ADVERSE_EVENT` | 🟠 Orange | fatigue, nausea, thrombocytopenia |
| `TRIAL_PHASE` | 🟢 Green | Phase III, Phase II |
| `BIOMARKER` | 🟣 Purple | BCL-2, mRNA, overall survival |
| `PATIENT_POP` | 🔵 Blue | 43,548 participants, adults aged 16+ |
| `PROCEDURE` | 🩷 Pink | intramuscular injection, randomized controlled trial |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/aakashmak/ClinicalNER.git
cd ClinicalNER
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `nemo_toolkit[nlp]` requires ~4GB disk space and downloads pretrained model weights on first run. A CUDA-compatible GPU is recommended but not required for demo purposes.

### 3. Run the app
```bash
streamlit run app.py
```

---

## 💻 Usage

```python
from src.ner_engine import ClinicalNEREngine

engine = ClinicalNEREngine(use_nemo=True)

text = """
A Phase III trial evaluated BNT162b2 administered as two 30 mcg doses.
Adverse events included fatigue and headache. BCL-2 biomarker expression
was assessed in 43,548 participants aged 16 years and older.
"""

entities = engine.predict(text)
summary  = engine.get_entity_summary(entities)
df       = engine.to_dataframe(entities)

print(summary)
# {
#   'TRIAL_PHASE': ['Phase III'],
#   'DRUG': ['BNT162b2'],
#   'DOSAGE': ['30 mcg'],
#   'ADVERSE_EVENT': ['fatigue', 'headache'],
#   'BIOMARKER': ['BCL-2'],
#   'PATIENT_POP': ['43,548 participants']
# }
```

---

## 🔬 NVIDIA NeMo Model Details

This project uses NeMo's **TokenClassificationModel** with a pretrained `ner_en_bert` checkpoint from NVIDIA NGC:

```python
import nemo.collections.nlp as nemo_nlp

model = nemo_nlp.models.TokenClassificationModel.from_pretrained("ner_en_bert")
```

**For domain-adaptive fine-tuning on clinical data:**
```python
# Fine-tune on CDISC/clinical NER dataset
model = nemo_nlp.models.TokenClassificationModel(cfg=config)
trainer.fit(model)
```

NeMo supports fine-tuning on custom BIO-tagged datasets in CoNLL format — enabling adaptation to CDISC terminology, clinical trial schemas, and proprietary pharmaceutical vocabularies.

---

## 📁 Project Structure

```
ClinicalNER/
├── app.py                  # Streamlit UI
├── src/
│   ├── __init__.py
│   └── ner_engine.py       # NeMo NER engine + fallback tagger
├── data/                   # Sample clinical texts (optional)
├── outputs/                # Exported CSV/JSON results
├── requirements.txt
└── README.md
```

---

## 🔭 Roadmap

- [ ] Fine-tune NeMo model on [BC5CDR](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/) biomedical NER dataset
- [ ] Add CDISC SDTM domain mapping layer
- [ ] Integrate NVIDIA NIM inference microservice for GPU-accelerated deployment
- [ ] Batch processing for 1000+ abstracts via CSV upload
- [ ] Add relation extraction (drug → adverse event linkage)

---

## 🏥 Use Cases

- **Pharmaceutical companies** — automated extraction from clinical study reports
- **CROs** — accelerate CDISC data mapping and validation
- **Medical literature mining** — structured data from PubMed/ClinicalTrials.gov
- **Pharmacovigilance** — adverse event signal detection from unstructured sources

---

## 👤 Author

**Aakash Mathivanan**
- MS Data Analytics, Northeastern University
- Senior Data Engineer @ Saama Technologies (AbbVie, Moderna, PPD)
- [GitHub](https://github.com/aakashmak) | [Email](mailto:aakashmak1809@gmail.com)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
