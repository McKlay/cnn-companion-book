---
hide:
    - toc
---

# Chapter 15: *Training Strategies and Fine-Tuning Pretrained CNNs*

> *“A good model trains well. A great model generalizes. The difference is in your training strategy.”*

---

## Why This Chapter Matters

Even the best CNN architectures can fail if:

* You train too long
* You train the wrong layers
* You don't handle data imbalance
* You mismatch inputs with pretrained expectations

This chapter will teach you how to:

* **Freeze**, **fine-tune**, and **retrain** CNNs correctly
* Apply **regularization** and **learning rate schedules**
* Handle **imbalanced datasets** the right way
* Recognize and respond to **overfitting vs underfitting**

Whether you're training from scratch or adapting ResNet to classify medical images, this chapter gives you **battle-tested practices** for *generalization-focused training*.

---

## 🔹 1. When to Fine-Tune vs Freeze

### 🔸 Base Layers vs Top Layers

* **Base layers**: Earlier convolutional blocks that detect general patterns (edges, corners, textures)
* **Top layers**: Deeper blocks and classifiers that detect task-specific patterns

---

### 🔸 Three Training Scenarios

| Strategy               | What It Does                                           | When to Use                                   |
| ---------------------- | ------------------------------------------------------ | --------------------------------------------- |
| **Feature Extraction** | Freeze all convolutional layers, train classifier only | Small custom dataset, fast prototyping        |
| **Fine-Tuning (Top)**  | Freeze early layers, train top conv + classifier       | Medium dataset, similar domain to ImageNet    |
| **Full Retraining**    | Train all layers                                       | Large dataset, significantly different domain |

---

### 🔸 PyTorch Implementation: Freezing Layers

```python
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze top layers
for param in model.classifier.parameters():
    param.requires_grad = True

model.eval()  # important for correct BatchNorm and Dropout behavior
```

### 🔸 TensorFlow Implementation: Freezing Layers

```python
# Freeze base model
base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add top layers
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(3)  # your class count
])
```

To unfreeze selectively:

```python
for layer in base_model.layers[-20:]:  # Unfreeze last 20 layers
    layer.trainable = True
```

---

## 🔹 2. Adapting Pretrained Models

### 🔸 Replace Output Layer

Most pretrained models end with Dense layers for 1000 ImageNet classes. You’ll need to:

* Replace the last Dense/Linear layer
* Match your dataset’s class count

#### PyTorch

```python
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

#### TensorFlow

```python
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes)
])
```

---

### 🔸 Use Adaptive Pooling for Any Input Size

CNNs expect fixed-size inputs (e.g., 224×224), but you can:

* Use `AdaptiveAvgPool2d((1, 1))` in PyTorch
* Use `GlobalAveragePooling2D()` in TensorFlow

These remove the dependence on fixed spatial dimensions.

---

### 🔸 Normalize Inputs to Match Model Expectation

If you use a pretrained ResNet or MobileNet:

* Match the **mean/std normalization**
* Use the correct **channel order** and **value range**

See Chapter 5 for full details.

---

## 🔹 3. Regularization Techniques

Regularization helps prevent overfitting.

### 🔸 Dropout

* Randomly drops neurons during training
* Use **after Dense layers**, not Convs
* Typical value: `0.3` to `0.5`

```python
nn.Dropout(0.5)  # PyTorch
layers.Dropout(0.5)  # TensorFlow
```

---

### 🔸 Weight Decay (L2 Regularization)

Applies penalty to large weights

| Framework  | How to Use                                                         |
| ---------- | ------------------------------------------------------------------ |
| PyTorch    | `optim.Adam(..., weight_decay=1e-4)`                               |
| TensorFlow | Add kernel regularizer: `Dense(..., kernel_regularizer=l2(0.001))` |

---

### 🔸 Handling Data Imbalance

#### 1. Class Weights in Loss

* Assign higher weight to underrepresented classes

```python
# PyTorch
weights = torch.tensor([1.0, 2.0, 0.5])  # adjust per class
criterion = nn.CrossEntropyLoss(weight=weights)
```

```python
# TensorFlow
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(..., loss=loss_fn, class_weight={0:1.0, 1:2.0, 2:0.5})
```

---

#### 2. Oversampling

* Duplicate rare samples
* Can be done manually or via `WeightedRandomSampler` in PyTorch

---

#### 3. Dataset Inspection

Always:

* Visualize sample counts per class
* Log confusion matrix during validation

---

## 🔹 4. Training Strategies for Generalization

### 🔸 Early Stopping

Stop training when validation stops improving

| PyTorch                     | TensorFlow                                             |
| --------------------------- | ------------------------------------------------------ |
| Manual via patience counter | `EarlyStopping(patience=3, restore_best_weights=True)` |

---

### 🔸 Learning Rate Schedules

| Strategy          | Purpose                           | PyTorch                                     | TensorFlow                       |
| ----------------- | --------------------------------- | ------------------------------------------- | -------------------------------- |
| StepLR            | Decays LR every N epochs          | `StepLR(optimizer, step_size=5, gamma=0.1)` | `LearningRateScheduler` callback |
| ReduceLROnPlateau | Reduce LR when val loss plateaus  | `ReduceLROnPlateau(...)`                    | `ReduceLROnPlateau(...)`         |
| Cosine Annealing  | Oscillates learning rate smoothly | `CosineAnnealingLR(...)`                    | Custom `LearningRateScheduler`   |

---

### 🔸 Gradual Unfreezing

In fine-tuning:

* Start with base frozen
* Unfreeze one block at a time
* Reduce LR when unfreezing

---

## 🔹 5. Recognizing Overfitting and Underfitting

### 🔸 Visual Clues from Loss/Accuracy Curves

| Symptom                     | Diagnosis             | Fix Suggestion                           |
| --------------------------- | --------------------- | ---------------------------------------- |
| High train acc, low val acc | **Overfitting**       | Add regularization, more data, dropout   |
| Flat train + val accuracy   | **Underfitting**      | Increase model capacity or training time |
| Val loss spikes upward      | **Training too long** | Use early stopping                       |

📌 Use `matplotlib` to plot:

* `train_loss`, `val_loss`, `train_acc`, `val_acc` vs epoch

---

## Framework Comparison Table

| Concept              | PyTorch                            | TensorFlow                          |
| -------------------- | ---------------------------------- | ----------------------------------- |
| Freeze layers        | `requires_grad = False`            | `layer.trainable = False`           |
| Replace output layer | `model.fc = nn.Linear(...)`        | `Dense(...)` on top of `base_model` |
| Adaptive pooling     | `nn.AdaptiveAvgPool2d((1,1))`      | `GlobalAveragePooling2D()`          |
| Weight decay         | `optimizer(..., weight_decay=...)` | `kernel_regularizer=l2(...)`        |
| Class weighting      | `CrossEntropyLoss(weight=...)`     | `class_weight={...}` in `fit()`     |
| Early stopping       | Manual or custom                   | Built-in `EarlyStopping` callback   |
| Gradual unfreeze     | Manual per parameter               | Manual per layer                    |

---

## Mini-Exercise

Fine-tune a pretrained ResNet50 to classify 3 new classes.

1. Load model with `include_top=False`
2. Add:

   * Global average pooling
   * Dense output layer
3. Freeze base
4. Train only top for 5 epochs
5. Then unfreeze last block
6. Add:

   * Dropout
   * L2 regularization
   * ReduceLROnPlateau
   * Early stopping
7. Plot train/val loss and accuracy
8. Identify if it overfits or underfits

**Bonus**: Try with both PyTorch and TensorFlow.

---
