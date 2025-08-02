# Entity-Aware Machine Translation (EA-MT) with NER

## Overview
This project implements an advanced Entity-Aware Machine Translation (EA-MT) system for the SemEval 2025 NER task. The goal is to translate English sentences into French while accurately identifying and translating named entities (NEs), including rare, ambiguous, or unknown entities. The system leverages multi-task learning with integrated Named Entity Recognition (NER) and a robust translation pipeline.

---

## Table of Contents
- [Project Motivation](#project-motivation)
- [Directory Structure](#directory-structure)
- [Data Format & Preparation](#data-format--preparation)
- [Model Architecture & Approach](#model-architecture--approach)
- [Entity-Aware Translation Pipeline](#entity-aware-translation-pipeline)
- [Handling Large Files & Git Best Practices](#handling-large-files--git-best-practices)
- [Setup & Usage](#setup--usage)
- [Acknowledgements](#acknowledgements)

---

## Project Motivation
Named entities are critical for accurate translation, especially when dealing with rare or ambiguous terms. Standard MT systems often fail to preserve or correctly translate these entities. Our approach integrates NER into the MT pipeline, ensuring entities are recognized, tagged, and translated with special care.

---

## Directory Structure
```
├── entity_aware_mt
│   ├── src
│   │   ├── baseline_translation.py
│   │   ├── check_gpu.py
│   │   ├── data_preparation.py
│   │   ├── entity_aware_translation.py
│   │   ├── finetune_placeholder_mt.py
│   │   ├── postprocess_entities.py
│   │   ├── predict_finetuned_mt.py
│   │   └── translate_placeholders.py
│   └── finetuned_placeholder_mt/  # (Ignored in git)
├── predictions.jsonl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Format & Preparation
- **Input:** English sentences in SemEval 2025 JSONL format, annotated with named entities.
- **Output:** French translations with entity-aware processing (entities tagged and translated).
- **Why?** JSONL is efficient for large-scale NLP tasks and preserves annotation structure for NER.
- **Script:** `data_preparation.py` parses and prepares data for model training and evaluation.

---

## Model Architecture & Approach
- **Multi-task Learning:**
  1. **NER Task:** Identifies entities in the source sentence.
  2. **Entity-Aware Translation:** Translates sentences, marking entities with special tags.
  3. **Entity Translation:** Translates the entities themselves, possibly using a specialized dictionary or sub-model.
- **Why?** Jointly training NER and MT improves translation accuracy for entities, especially rare or ambiguous ones.

---

## Entity-Aware Translation Pipeline
1. **NER Tagging** (`entity_aware_translation.py`):
   - Detects entities in the input sentence.
2. **Placeholder Translation** (`translate_placeholders.py`):
   - Replaces entities with placeholders to ensure correct context translation.
3. **Finetuned MT Prediction** (`predict_finetuned_mt.py`):
   - Uses a finetuned MT model to translate the sentence with placeholders.
4. **Entity Translation** (`finetune_placeholder_mt.py`):
   - Translates entities separately, handling rare/ambiguous cases.
5. **Postprocessing** (`postprocess_entities.py`):
   - Replaces placeholders in the translated sentence with the translated entities.
6. **Evaluation** (`eval.ipynb`):
   - Jupyter notebook for evaluating translation and NER performance.

**Why this pipeline?**
- Separating entity translation avoids errors from the MT model on rare/unknown entities.
- Placeholder mechanism ensures context is preserved while maintaining entity fidelity.

---

## Handling Large Files & Git Best Practices
- **Problem:** Model checkpoints and large binary files (e.g., `.pt`, `.safetensors`) exceeded GitHub's 100MB limit.
- **Solution:**
  - Updated `.gitignore` to exclude all large model and checkpoint files:
    ```
    *.pt
    *.safetensors
    *.bin
    checkpoint-*/
    entity_aware_mt/finetuned_placeholder_mt/
    ```
  - Used `git filter-repo` to remove large files from git history:
    ```
    git filter-repo --strip-blobs-bigger-than 100M
    ```
  - (Optional) For future large files, consider using [Git LFS](https://git-lfs.github.com/).

**Why?**
- Keeps the repository lightweight and compliant with GitHub's file size policies.
- Ensures only code and small data files are versioned.

---

## Setup & Usage

### 1. Clone the repository
```sh
git clone https://github.com/Parth6120/NER_SemEval_2025.git
cd NER_SemEval_2025
```

### 2. Install dependencies
```sh
pip install -r requirements.txt
```

### 3. Prepare Data
- Place your SemEval 2025 JSONL files in the appropriate directory.
- Use `data_preparation.py` to preprocess the data.

### 4. Train or Finetune Models
- Use `finetune_placeholder_mt.py` for finetuning the MT model with entity placeholders.

### 5. Run Translation Pipeline
- Use the scripts in `src/` as described above to process, translate, and evaluate your data.

---

## Acknowledgements
- SemEval 2025 NER Task organizers
- HuggingFace Transformers for model architectures
- Inspiration from recent research in entity-aware and multi-task MT

---

## References
- [GitHub LFS](https://git-lfs.github.com/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [SemEval](https://semeval.github.io/)

---

For any questions or suggestions, please open an issue or contact the maintainer.