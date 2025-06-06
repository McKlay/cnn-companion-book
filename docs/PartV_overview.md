---
hide:
    - toc
---

# Part V – Inference, Evaluation, and Visual Debugging

> *“Once the model is trained, the real-world test begins. Inference is the moment your model meets reality.”*

---

## Why This Part Matters

Training a CNN isn’t the final goal. What really matters is what happens **after training**:

* Does it make correct predictions on unseen data?
* Does it behave consistently across platforms?
* Can you debug it when it fails?

Most real-world issues occur **after** training:

* Input mismatch during inference
* Dropout or BatchNorm behaving incorrectly
* Unexplainable results that need visualization

This part of the book equips you to **evaluate models confidently and debug them visually.**

---

## What You’ll Master in This Part

* Understand and use the correct training/evaluation modes
* Handle **Dropout** and **BatchNorm** properly during inference
* Perform **accurate predictions** in real-time systems
* Visualize **feature maps** and **filters** to interpret what the model is learning
* Build intuition for CNN internals using visual tools

---

## Chapter Breakdown

| Chapter | Title                                  | What You’ll Learn                                                      |
| ------- | -------------------------------------- | ---------------------------------------------------------------------- |
| **16**  | *Train vs Eval Mode*                   | Understand mode switching, dropout behavior, and inference consistency |
| **17**  | *Visualizing Feature Maps and Filters* | Peek inside CNNs to see what filters are activating and when           |

---

## Key Problems This Part Solves

| Problem                                    | Solution You'll Learn                            |
| ------------------------------------------ | ------------------------------------------------ |
| Model behaves differently at inference     | Use `model.eval()` or `training=False` properly  |
| Accuracy drops on deployment               | Check normalization, dropout, batchnorm behavior |
| Can't explain model predictions            | Visualize filters and activations                |
| Confusion between logits and probabilities | Confirm mode, output, and softmax usage          |
| Debugging performance bottlenecks          | Test inference-only mode with no gradients       |

---

## Tools You’ll Be Using

| Tool                           | Description                               |
| ------------------------------ | ----------------------------------------- |
| `model.eval()` / `.train()`    | Control dropout/batchnorm in PyTorch      |
| `training=True/False`          | Control training/inference in TensorFlow  |
| `torch.no_grad()`              | Disable gradients for inference           |
| Keras `Model(inputs, outputs)` | Create sub-models for visualization       |
| PyTorch forward hooks          | Inspect layer outputs during forward pass |
| `matplotlib`, `seaborn`        | Plot activation maps and filters          |

---

## What You’ll Be Able To Do After This Part

* Switch between training and inference with complete confidence
* Ensure consistent behavior between training and deployment environments
* Analyze **intermediate activations** in CNNs
* Debug failing models visually
* Understand how convolution filters “light up” for different image regions

---

## Real-World Relevance

> In production, bugs often stem not from your training pipeline—but from incorrect inference behavior or misunderstood activations.

You’ll encounter scenarios like:

* A model doing well in training but failing in deployment due to `Dropout` still being active
* A medical image classifier that seems to focus on **text labels** in X-rays—visible through feature maps
* A classifier always choosing one class—fixable only after visualizing activation saturation

This part helps you **trust, verify, and explain** what your CNN is doing under the hood.

---
