---
hide:
    - toc
---

# Chapter 11: *Model Summary and Parameter Inspection*

> *“Understanding your model is like reading blueprints before construction. Don’t train it blind—inspect, analyze, and optimize.”*

---

## Why This Chapter Matters

You’ve built a CNN and defined its forward flow. But now you need to:

* Check if the architecture matches your expectations
* Inspect layer shapes and total parameter counts
* **Freeze or unfreeze layers** for transfer learning
* Load/save weights or extract specific layer outputs

This chapter helps you understand:

* How to **summarize your model**
* How to **access weights and parameters**
* How to **manage layers for training or inference**

You’ll learn to treat models not as magic boxes—but as transparent, inspectable systems.

---

## Conceptual Breakdown

### 🔹 What to Inspect in a Model

| Property            | Why it Matters                             |
| ------------------- | ------------------------------------------ |
| Layer names/types   | Ensure architecture is correct             |
| Output shapes       | Catch shape mismatches early               |
| Total parameters    | Know model size and overfitting risk       |
| Trainable vs frozen | Required for transfer learning/fine-tuning |
| Weight values       | For debugging, initialization checks       |

---

### 🔹 Freezing vs Unfreezing Layers

**Freezing a layer** means its weights won’t update during training (used in transfer learning).
**Unfreezing** means allowing gradient flow again.

📌 Freeze base layers → train top layers only → unfreeze gradually.

---

## PyTorch Implementation

### 🔸 Model Summary

Use the `torchsummary` package (or print manually):

```bash
pip install torchsummary
```

```python
from torchsummary import summary
import torch

model = ConvClassifier()
summary(model, input_size=(3, 224, 224))
```

### 🔸 Inspect Parameters

```python
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")
```

### 🔸 Freeze Layers

```python
for param in model.features.parameters():
    param.requires_grad = False
```

To unfreeze later:

```python
for param in model.features.parameters():
    param.requires_grad = True
```

### 🔸 Save/Load Weights

```python
torch.save(model.state_dict(), "model_weights.pth")

model.load_state_dict(torch.load("model_weights.pth"))
model.eval()  # Set to inference mode
```

---

## TensorFlow Implementation

### 🔸 Model Summary

```python
model = ConvClassifier()
model.build(input_shape=(None, 224, 224, 3))
model.summary()
```

### 🔸 Inspect Weights

```python
for layer in model.layers:
    print(layer.name, layer.trainable)
    for weight in layer.weights:
        print(f"  {weight.name} - shape: {weight.shape}")
```

### 🔸 Freeze Layers

```python
for layer in model.layers:
    layer.trainable = False  # freeze
```

Unfreeze:

```python
for layer in model.layers:
    layer.trainable = True
```

### 🔸 Save/Load Weights

```python
model.save_weights("model_checkpoint.h5")

# Reload weights into the same architecture
model.load_weights("model_checkpoint.h5")
```

---

## PyTorch vs TensorFlow Parameter Access

| Task              | PyTorch                       | TensorFlow                |
| ----------------- | ----------------------------- | ------------------------- |
| Get all weights   | `model.parameters()`          | `model.weights`           |
| Get named weights | `model.named_parameters()`    | `layer.weights` per layer |
| Freeze training   | `param.requires_grad = False` | `layer.trainable = False` |
| Save weights      | `torch.save(state_dict)`      | `model.save_weights()`    |
| Load weights      | `model.load_state_dict(...)`  | `model.load_weights()`    |

---

## Model Modes: Train vs Eval

| Mode          | PyTorch                     | TensorFlow                      |
| ------------- | --------------------------- | ------------------------------- |
| Training      | `model.train()`             | `training=True` in `call()`     |
| Inference     | `model.eval()`              | `training=False` in `call()`    |
| Dropout/BNorm | Behave differently in modes | Same applies in both frameworks |

Always remember:

* Use **`model.eval()`** in PyTorch during inference (turns off dropout, uses running stats in BatchNorm).
* In TensorFlow, use `training=False` explicitly in `call()`.

---

## Mini-Exercise

1. Build a small CNN for 3-class classification.
2. Print full model summary.
3. Freeze all convolutional layers.
4. Confirm only `Linear` / `Dense` layers are trainable.
5. Save and reload weights.
6. Switch between train/eval modes and print dropout effect.

**Bonus**: Write a utility function that counts:

* Total parameters
* Trainable parameters
* Frozen parameters

---

