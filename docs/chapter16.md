---
hide:
    - toc
---

# Chapter 16: *Train vs Eval Mode*

> *“A model is like a chameleon—it changes behavior depending on whether it’s training or being tested. Know when it’s learning and when it should just perform.”*

---

## Why This Chapter Matters

When you're debugging or deploying CNNs, switching between **training** and **evaluation (inference)** modes is crucial. Two components in particular behave differently depending on the mode:

* **Dropout**: Randomly disables neurons during training to prevent overfitting
* **Batch Normalization**: Uses running mean/variance during inference instead of batch statistics

If you forget to switch to inference mode:

* The model will behave unpredictably
* Validation accuracy will look unstable
* Final test performance may drop drastically

This chapter helps you:

* Understand the mechanics of mode switching
* Avoid silent bugs in inference
* Use PyTorch and TensorFlow's tools for correct evaluation behavior

---

## Conceptual Breakdown

### 🔹 The Two Modes

| Mode      | Description                                        | When to Use                               |
| --------- | -------------------------------------------------- | ----------------------------------------- |
| **Train** | Active learning: dropout + batchnorm use live data | During training phase                     |
| **Eval**  | Inference mode: deterministic behavior             | During validation, testing, or deployment |

---

### 🔹 Layers That Behave Differently

| Layer Type    | Training Mode Behavior                     | Eval Mode Behavior                         |
| ------------- | ------------------------------------------ | ------------------------------------------ |
| **Dropout**   | Randomly zeroes out neurons per batch      | Skipped entirely (no dropout at inference) |
| **BatchNorm** | Uses current batch stats for normalization | Uses running (moving average) stats        |

---

## PyTorch Implementation

### 🔸 Switching Modes

```python
model.train()  # Activates Dropout + BatchNorm (training mode)
...
model.eval()   # Freezes Dropout + BatchNorm (inference mode)
```

### 🔸 Disabling Gradients During Inference

```python
model.eval()
with torch.no_grad():
    outputs = model(images)
```

Why `torch.no_grad()`?

* Saves memory
* Speeds up inference
* Ensures gradients are not computed (and no backward graph is tracked)

---

### 🔸 Validation Code Snippet

```python
model.eval()
val_loss = 0.0
correct = 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
```

---

## TensorFlow Implementation

### 🔸 Mode Switching with `training=True/False`

In TensorFlow, mode is passed explicitly to the model during `call()`:

```python
preds = model(images, training=True)   # Train mode
preds = model(images, training=False)  # Inference mode
```

You must use this flag correctly in:

* Manual training loops
* Custom `Model` subclasses

---

### 🔸 Example: Manual Validation

```python
# Validation step
for x_batch_val, y_batch_val in val_dataset:
    val_logits = model(x_batch_val, training=False)  # eval mode
    val_loss = loss_fn(y_batch_val, val_logits)
```

### 🔸 With `model.fit()`

Keras handles mode switching automatically when using `model.fit()` and `model.evaluate()`. But in custom training loops, it’s manual.

---

### 🔸 Model Summary Check

```python
model.summary()  # Always available
print([layer.trainable for layer in model.layers])  # Check trainable flags
```

---

## Common Mistakes and How to Avoid Them

| Mistake                                   | Consequence                                | Fix                                              |
| ----------------------------------------- | ------------------------------------------ | ------------------------------------------------ |
| Forgetting `.eval()` in PyTorch           | Dropout and BN are active during inference | Call `model.eval()` before validation or testing |
| Forgetting `training=False` in TensorFlow | Model behaves like it's still training     | Pass `training=False` explicitly in calls        |
| Not using `torch.no_grad()`               | Higher memory usage during inference       | Wrap evaluation in `with torch.no_grad()`        |
| Logging wrong metrics                     | Misinterpreted validation accuracy         | Ensure eval mode + no gradient during val        |

---

## Framework Comparison Table

| Concept                    | PyTorch                 | TensorFlow                                   |
| -------------------------- | ----------------------- | -------------------------------------------- |
| Activate training mode     | `model.train()`         | `training=True` in `model(x, training=True)` |
| Activate evaluation mode   | `model.eval()`          | `training=False`                             |
| Disable gradient tracking  | `with torch.no_grad():` | Automatic in `fit()` / use tape manually     |
| BatchNorm/Dropout behavior | Respects mode setting   | Respects `training` flag                     |
| Manual control needed      | Yes                     | Yes for custom loops, no for `fit()`         |

---

## Mini-Exercise

Implement the following for a simple CNN classifier:

1. Write a validation loop in **both PyTorch and TensorFlow**
2. In **PyTorch**, explicitly switch between `.train()` and `.eval()`, and use `torch.no_grad()`
3. In **TensorFlow**, pass `training=True` or `False` depending on the phase
4. Compare the model output with dropout active vs inactive

**Bonus**: Log memory usage during inference with and without gradient tracking

---

##  What You Can Now Do

* Evaluate models with consistent accuracy and no randomness
* Avoid common dropout and batchnorm bugs
* Use inference mode to:

  * Save memory
  * Improve speed
  * Ensure deployment stability

---