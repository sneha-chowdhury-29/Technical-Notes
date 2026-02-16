# Speech Emotion Recognition using Wav2Vec2

## Technical Learnings & Engineering Notes

---

# Project Objective

Build an 8-class Speech Emotion Recognition (SER) system using a transformer-based speech model (Wav2Vec2) fine-tuned on the RAVDESS dataset.

The goal was not just to train a model, but to:

* Handle raw waveform input
* Build a reproducible training pipeline
* Implement evaluation metrics correctly
* Support inference on custom audio
* Convert notebook experimentation into a clean GitHub project

---

# Model Choice: Why Wav2Vec2?

Instead of traditional MFCC + CNN/LSTM pipelines, this project used:

```
facebook/wav2vec2-base
```

### Why?

* Pretrained on large-scale unlabeled speech
* Learns contextual acoustic representations
* Avoids handcrafted features
* Performs strongly on downstream speech tasks

Key takeaway:

> Pretrained self-supervised speech models significantly reduce feature engineering effort and improve performance.

---

# Data Engineering Lessons

## RAVDESS Label Parsing

Emotion labels were encoded inside filenames.

Lesson:

> Always validate label parsing logic early. Incorrect mapping silently ruins performance.

---

## Variable-Length Audio

Speech samples have different durations.

Initial issue:

* Trainer tried stacking tensors directly
* Caused runtime crashes

Solution:
Implemented custom dynamic padding via a `data_collator`.

```python
def data_collator(features):
    input_values = [f["input_values"] for f in features]
    labels = torch.tensor([f["labels"] for f in features])

    batch = processor(
        input_values,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )

    batch["labels"] = labels
    return batch
```

Lesson:

> Raw audio requires dynamic padding before batching. Transformers don’t handle waveform padding automatically.

---

# Environment & Dependency Debugging

This project involved significant environment alignment challenges.

---

## Python Version Compatibility

Initial attempt used Python 3.14.

Issues:

* `datasets` + `pyarrow` errors
* HuggingFace backend crashes

Solution:

* Downgraded to Python 3.11
* Created virtual environment
* Registered new Jupyter kernel

Lesson:

> ML ecosystems lag behind latest Python releases. Use stable versions.

---

## Transformers + Accelerate Version Conflicts

Issues encountered:

* `evaluation_strategy` vs `eval_strategy`
* Callback handler crashes
* `accelerate` version mismatch
* TensorBoard integration failures

Final stable configuration:

```
transformers == 4.40.2
accelerate == 0.31.0
```

Lesson:

> Transformers + Accelerate must be version-aligned. Random upgrades break Trainer pipelines.

---

## Apple Silicon (MPS) Issues

Problems:

* `pin_memory=True` not supported
* CPU vs MPS tensor mismatch
* Stale device memory in notebook

Fixes:

* `dataloader_pin_memory=False`
* Explicit `model.to(device)`
* Kernel restarts to clear stale tensors

Lesson:

> Device mismatches are a common source of runtime errors. Always confirm `model.parameters()` device.

---

# 5️Training Insights

## Initial Training (3 epochs)

* Accuracy ≈ 69%
* Underfitting observed

## Improved Configuration

* Epochs: 8
* Learning rate: 2e-5
* Batch size: 4

Final performance:

* Accuracy ≈ 91%
* F1 Macro ≈ 0.91
* F1 Weighted ≈ 0.91

Lesson:

> Speech models often require more fine-tuning epochs than expected. Early stopping too soon can underfit.

---

# Evaluation Strategy

Implemented:

* Accuracy
* Weighted F1
* Macro F1

Why both F1 scores?

* Weighted F1 → reflects dataset distribution
* Macro F1 → treats all classes equally

Confusion matrix analysis showed:

* Strong diagonal dominance
* Minor confusion between emotionally similar classes
* Balanced class performance

Lesson:

> Always evaluate both macro and weighted F1 in multi-class classification.

---

# Inference Engineering

Built CLI-based inference:

```bash
python inference.py --audio path/to/file.wav
```

Pipeline:

1. Load model
2. Load processor
3. Preprocess audio
4. Forward pass
5. Argmax over logits
6. Map ID to emotion

Lesson:

> Separate training and inference logic. This makes the project deployable.

---

# Project Structure Lessons

Final structure:

```
train.py
inference.py
SER_pipeline.ipynb
README.md
```

Decision:

* Keep notebook for experimentation
* Use scripts for reproducibility

Lesson:

> Clean script-based training improves credibility compared to notebook-only projects.

---

# Limitations

* Dataset is studio-quality (limited real-world robustness)
* No data augmentation used
* No cross-dataset validation
* Real-time microphone inference not implemented

---

# Future Improvements

* Add SpecAugment
* Experiment with HuBERT
* Freeze feature extractor initially
* Perform hyperparameter sweep
* Deploy via Streamlit

---

# Overall Learnings

This project reinforced that:

* Environment management is as important as modeling
* Version alignment prevents subtle runtime failures
* Device handling matters in speech ML
* Modularization improves clarity
* Fine-tuning pretrained speech models is powerful but fragile

---

# Conclusion

This project evolved from notebook experimentation into a reproducible transformer-based speech classification system achieving ~91% accuracy on 8-class emotion recognition.

Beyond modeling, the primary value came from:

* Debugging ML infrastructure
* Understanding Trainer internals
* Designing robust data collation
* Managing device and dependency conflicts

