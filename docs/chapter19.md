---
hide:
    - toc
---

# Chapter 19: *Common Errors and How to Debug Them*

> *“The most dangerous bugs are silent. CNNs won’t throw exceptions when they’re wrong—they’ll just smile and mispredict.”*

---

## Why This Chapter Matters

After training your CNN and setting up a beautiful inference pipeline, everything looks good—but:

* The model always predicts the same class
* Accuracy is much lower than expected
* It fails on real-world images even if training went fine

These are **not training bugs**. They’re **systemic failures** often due to:

* Data leakage
* Input misalignment
* Normalization errors
* Inconsistent shapes
* Forgotten mode switching

This chapter equips you with a **checklist and mindset** to debug confidently.

---

## Conceptual Breakdown

### 🔍 Where Most Bugs Hide

| Bug Type             | Example                                           | Detection Strategy                                  |
| -------------------- | ------------------------------------------------- | --------------------------------------------------- |
| **Normalization**    | Wrong mean/std or none at inference               | Compare input histograms before and after           |
| **Shape mismatch**   | Model expects \[3, 224, 224], gets \[1, 256, 256] | Print `.shape` at each step                         |
| **Wrong eval mode**  | Model trained well but fails validation           | Ensure `model.eval()` or `training=False`           |
| **Data leakage**     | Same image appears in train and test set          | Check file paths, cross-fold splits                 |
| **One-class output** | Predicts class 0 for everything                   | Check for class imbalance, final layer, softmax use |

---

## PyTorch Debugging Checklist

```python
# 1. Check model mode
print(model.training)  # Should be False during inference

# 2. Check input shape
print(image_tensor.shape)  # Should be [1, 3, 224, 224]

# 3. Check normalization
print(image_tensor.min(), image_tensor.max())  # Should be ~-1 to 1

# 4. Visual sanity check
import matplotlib.pyplot as plt
plt.imshow(image_tensor.squeeze().permute(1, 2, 0).numpy())
```

---

## TensorFlow Debugging Checklist

```python
# 1. Model mode
logits = model(img, training=False)

# 2. Shape check
print(img.shape)  # Should be (1, 224, 224, 3)

# 3. Normalization
print(img.min().numpy(), img.max().numpy())  # Depending on preprocess_input()

# 4. Show input
plt.imshow(img[0] / 2 + 0.5)  # Undo [-1,1] normalization for visualization
```

---

## 🛠 Common CNN Errors & Fixes

| Symptom                               | Cause                                              | Fix                                                          |
| ------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| Always predicting one class           | Class imbalance, untrained head, no softmax        | Use weighted loss, check class distribution, inspect logits  |
| Very low accuracy at inference        | Wrong normalization                                | Match training transforms exactly                            |
| Crash on inference                    | Missing batch dim or float32 type                  | Use `.unsqueeze(0)` and convert to `float32`                 |
| Fails on grayscale / RGBA image       | Expecting 3-channel RGB                            | Convert image `.convert("RGB")`                              |
| Prediction unstable during validation | Forgot `.eval()` or `no_grad()`                    | Call `model.eval()` and use `torch.no_grad()`                |
| Shape mismatch in pretrained model    | Mismatch in input resolution or output class count | Resize input and adapt output layer                          |
| Test image looks weird                | Bad rescale or wrong channel order                 | Visualize before and after transforms                        |
| Weird outputs in browser/mobile       | Tensor converted incorrectly to JS or HTML format  | Normalize properly and ensure channels/byte values are valid |

---

## Visual Debugging Techniques

### 1. Visualize Inputs Before and After Transform

```python
# PyTorch
plt.subplot(1,2,1); plt.imshow(PIL_img)
plt.subplot(1,2,2); plt.imshow(tensor.permute(1, 2, 0).numpy())

# TensorFlow
plt.imshow(img_tensor[0] / 2 + 0.5)  # If normalized to [-1, 1]
```

### 2. Log Confidence Scores

```python
# PyTorch
probs = torch.softmax(outputs, dim=1)
print(probs.topk(3))  # Top-3 prediction scores

# TensorFlow
probs = tf.nn.softmax(preds).numpy()
print(np.argsort(probs[0])[-3:][::-1])  # Top-3 classes
```

### 3. Debug Batch Processing

* Ensure same shape per sample
* All batches should be same dtype
* Check shuffling order

---

## Defensive Programming Tips

| Strategy                                 | Benefit                                  |
| ---------------------------------------- | ---------------------------------------- |
| `assert image.shape == (1, 3, 224, 224)` | Prevents hidden shape bugs               |
| `assert img.dtype == torch.float32`      | Ensures model receives valid input       |
| Logging before/after each step           | Makes silent bugs visible                |
| Try inference on one image manually      | Removes complexity, isolates the problem |
| Unit test your transforms                | Catch errors early                       |

---

## Framework Comparison Table

| Debug Task               | PyTorch                        | TensorFlow / Keras                       |
| ------------------------ | ------------------------------ | ---------------------------------------- |
| Check training/eval mode | `model.training`               | Explicit `training=False` flag           |
| Visualize input          | `permute(1,2,0)` on tensor     | `/ 2 + 0.5` if normalized to \[-1,1]     |
| Print prediction scores  | `softmax(outputs, dim=1)`      | `tf.nn.softmax()`                        |
| One-image inference      | `unsqueeze(0)` and `no_grad()` | `np.expand_dims()` and `model.predict()` |
| Inspect model layers     | `print(model)` or `summary()`  | `model.summary()`                        |

---

## Mini-Exercise

Try this on your own CNN project:

1. Pick an image from your test set
2. Run it through your full pipeline
3. Log:

   * Input shape, dtype
   * Image pixel range before/after preprocessing
   * Top-3 predicted classes with confidence
4. Visualize:

   * Input image
   * Preprocessed tensor
   * Activation maps (from Chapter 17)
5. Bonus:

   * Temporarily insert an invalid image (grayscale, wrong shape) and handle it gracefully

---

## 🔚 Final Tips: The Debugging Mindset

* **Assume nothing**—even if training went perfectly
* **Print and plot often**
* **Start small**: one image, one batch
* **Compare to known-good outputs** (reference image → known prediction)
* **Trace the full pipeline**: from raw input to final prediction

---

## What You Can Now Do

* Build sanity-checked pipelines that won’t fail silently
* Fix models that seem broken but are just misconfigured
* Gain **confidence and trust** in your model’s predictions
* Catch and fix **systemic bugs** before they go live

---