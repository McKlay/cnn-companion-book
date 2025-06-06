---
hide:
    - toc
---

# Chapter 12: *Building Your First CNN: Patterns and Pitfalls*

> *“Good CNNs don’t come from just stacking layers—they come from knowing why you stack them.”*

---

## Why This Chapter Matters

You now understand:

* How images become tensors,
* How layers like Conv2D, Pooling, and BatchNorm work,
* How to write clean `forward()` or `call()` functions,
* How to inspect models and control parameters.

This chapter helps you:

* Design your own architecture **from scratch**
* Follow proven **CNN patterns**
* Avoid **common architectural mistakes**
* Set yourself up for **scaling later to deeper or pretrained models**

It's your first step toward becoming a deep learning architect—not just a user.

---

## Conceptual Breakdown

### 🔹 What Makes a "Good" CNN?

| Good CNN Design Has…                     | Why It Matters                                  |
| ---------------------------------------- | ----------------------------------------------- |
| Clear separation of feature + classifier | Easier to extend or replace sections            |
| Progressive increase in filter count     | Helps extract richer features deeper in network |
| Downsampling at reasonable intervals     | Balances spatial resolution vs computation      |
| Non-linearities + normalization          | Improves gradient flow, training stability      |
| Proper flattening before dense layers    | Ensures correct classifier input shape          |

---

### 🔹 Classic CNN Design Patterns

#### 🧱 LeNet (1998)

* Small filters (5×5), low depth
* MaxPooling for downsampling
* Fully connected at the end

```text
INPUT → Conv → ReLU → Pool → Conv → ReLU → Pool → FC → FC → Softmax
```

#### 🧱 Mini-VGG Style

* Use **stacks** of 3×3 Conv layers before pooling
* Double filters after each pooling
* No FC until final layers

```text
INPUT → [Conv → ReLU] x2 → Pool → [Conv → ReLU] x2 → Pool → FC → Softmax
```

📌 Rule of thumb: **Double filters, halve resolution** after each pool

---

### 🔹 Choosing Filter Sizes

| Kernel Size | Best For                               |
| ----------- | -------------------------------------- |
| 1×1         | Reducing/increasing channel depth      |
| 3×3         | Most common, efficient pattern capture |
| 5×5         | Broader patterns, but costlier         |

📌 Stack 2× 3×3 layers instead of one 5×5 (same receptive field, fewer params)

---

### 🔹 When to Use Pooling

* Use **MaxPooling2D** or **stride=2** Conv2D to downsample
* Common after 1 or 2 Conv blocks
* Helps reduce computation and adds invariance to translation

📌 Avoid pooling too early—keep spatial detail in early layers

---

### 🔹 Flattening Correctly

* PyTorch: `.view(x.size(0), -1)` or `nn.Flatten()`
* TensorFlow: `Flatten()` layer

You can also use `AdaptiveAvgPool2d((1, 1))` or `GlobalAveragePooling2D()` to remove dependence on input image size.

---

### 🔹 Common Mistakes to Avoid

| Mistake                               | Consequence                                  |
| ------------------------------------- | -------------------------------------------- |
| Forgetting `.view()` / `.Flatten()`   | Shape error in Linear/Dense layer            |
| Pooling too early or too often        | Loss of spatial detail, underfitting         |
| Too few filters                       | Not enough capacity to learn visual patterns |
| Mismatched shapes at classifier input | Crash at final FC layer                      |
| No normalization or activation        | Poor learning and convergence                |

---

## 💻 PyTorch: Build a Clean MiniCNN

```python
import torch
import torch.nn as nn

class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),  # assuming input is 224×224
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

---

## 🧪 TensorFlow: Equivalent MiniCNN

```python
import tensorflow as tf
from tensorflow.keras import layers, models

class MiniCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool1 = layers.MaxPooling2D()

        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D()

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.out = layers.Dense(3)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        return self.out(x)
```

---

## Framework Comparison Table

| Element                  | PyTorch                               | TensorFlow                              |
| ------------------------ | ------------------------------------- | --------------------------------------- |
| Conv + ReLU + Pool block | `nn.Sequential()` + `nn.Conv2d`, etc. | `layers.Conv2D` + `ReLU` + `MaxPooling` |
| Flatten + Dense          | `nn.Flatten()` + `nn.Linear`          | `Flatten()` + `Dense()`                 |
| Dropout in training      | Auto-disabled in `eval()` mode        | Manual: `training=True` in `call()`     |
| Global pooling           | `AdaptiveAvgPool2d((1, 1))`           | `GlobalAveragePooling2D()`              |

---

## Mini-Exercise

Design a CNN for CIFAR-10 (input: 32×32×3):

1. Stack 3 Conv2D layers with increasing filters (e.g., 32 → 64 → 128)
2. Add ReLU + MaxPool after every 2 layers
3. Use Global Average Pooling before Dense
4. Use Dropout to prevent overfitting
5. Output 10 classes

**Bonus**: Replace MaxPool2D with `stride=2` Conv2D and compare performance.

---