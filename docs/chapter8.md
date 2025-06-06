---
hide:
    - toc
---

# Chapter 8: *Understanding CNN Layers*

> *“Every filter is a lens. Every layer is a language. A CNN doesn’t just see—it interprets.”*

---

## Why This Chapter Matters

A Convolutional Neural Network is more than a stack of layers—it’s a **hierarchy of abstractions.** With each convolution, pooling, and activation, your model goes from low-level pixels to high-level semantics:

* Edge → Shape → Texture → Object

But to design effective CNNs (and debug them), you need to understand how each layer **transforms the input.**

This chapter walks you through:

* What each major CNN layer does
* How it changes shape, depth, and meaning
* How to implement and visualize these layers in PyTorch and TensorFlow

You’ll finally understand **why a 224×224×3 image turns into a 7×7×512 feature map.**

---

## Conceptual Breakdown

### 🔹 The Core CNN Layer Types

| Layer                 | Function                                                     |
| --------------------- | ------------------------------------------------------------ |
| **Conv2D**            | Applies a filter/kernel over spatial regions                 |
| **Activation (ReLU)** | Adds non-linearity so the network can learn complex patterns |
| **BatchNorm**         | Normalizes activations to stabilize training                 |
| **Pooling**           | Reduces spatial size while keeping key features              |
| **Dropout**           | Prevents overfitting by randomly dropping activations        |
| **Fully Connected**   | Maps final features to output classes                        |

---

### 🔹 Convolution Layer: `Conv2D`

* Uses a **kernel** (e.g., 3×3) that slides across the image
* Performs **element-wise multiplications** and adds up the result
* Outputs a **feature map**

📌 A convolution layer doesn’t see the entire image—it sees a *window*. As we stack layers, the *receptive field* grows.

**Key parameters:**

* `in_channels`: number of input feature channels
* `out_channels`: number of filters (i.e., output channels)
* `kernel_size`: size of each filter (e.g., 3×3)
* `stride`: how much the filter moves per step
* `padding`: how edges are handled (valid vs same)

---

### 🔹 Pooling Layer: `MaxPool2D`, `AvgPool2D`

* **Downsamples** feature maps (e.g., from 32×32 → 16×16)
* Keeps strongest signals (MaxPooling) or averages regions (AvgPooling)
* Reduces computation and helps detect patterns invariant to position

---

### 🔹 Batch Normalization

* Normalizes output of a layer to have **zero mean, unit variance**
* Stabilizes training, allows for higher learning rates
* Applied **after convolution, before activation**

---

### 🔹 Activation Functions: ReLU and Beyond

| Activation | Formula          | Purpose                    |
| ---------- | ---------------- | -------------------------- |
| ReLU       | `max(0, x)`      | Introduces non-linearity   |
| Leaky ReLU | `max(αx, x)`     | Keeps small negative slope |
| Sigmoid    | `1 / (1 + e^-x)` | Squeezes to \[0, 1]        |

📌 Most modern CNNs use **ReLU** for its simplicity and efficiency.

---

### 🔹 Fully Connected (Dense) Layers

After several convolution + pooling blocks, the feature map is **flattened** into a vector and passed through one or more `Linear` (PyTorch) or `Dense` (TF) layers.

* Used to **classify** based on the features extracted earlier
* Last layer’s size = number of classes

---

## PyTorch Implementation

Let’s build a simple Conv → ReLU → Pool block:

```python
import torch.nn as nn

cnn_block = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),  # [B, 3, 224, 224] → [B, 16, 224, 224]
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)  # [B, 16, 224, 224] → [B, 16, 112, 112]
)
```

A full model:

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 10)  # assuming input was 224x224
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

---

## TensorFlow Implementation

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(16, (3, 3), padding='same', input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(32, (3, 3), padding='same'),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(10)
])
```

---

## How Shapes Change

| Operation       | PyTorch Shape Change                 | TensorFlow Shape Change              |
| --------------- | ------------------------------------ | ------------------------------------ |
| Conv2D          | `[B, C_in, H, W] → [B, C_out, H, W]` | `[B, H, W, C_in] → [B, H, W, C_out]` |
| MaxPool2D (2×2) | `[B, C, H, W] → [B, C, H/2, W/2]`    | `[B, H, W, C] → [B, H/2, W/2, C]`    |
| Flatten         | `[B, C, H, W] → [B, C×H×W]`          | `[B, H, W, C] → [B, H×W×C]`          |

---

## Framework Comparison Table

| Layer             | PyTorch                    | TensorFlow                    |
| ----------------- | -------------------------- | ----------------------------- |
| Convolution       | `nn.Conv2d(in, out, k)`    | `layers.Conv2D(filters, k)`   |
| Pooling           | `nn.MaxPool2d(k)`          | `layers.MaxPooling2D(k)`      |
| BatchNorm         | `nn.BatchNorm2d(channels)` | `layers.BatchNormalization()` |
| Activation (ReLU) | `nn.ReLU()` or `F.relu()`  | `layers.ReLU()` or inline     |
| Fully Connected   | `nn.Linear(in, out)`       | `layers.Dense(units)`         |
| Flatten           | `nn.Flatten()`             | `layers.Flatten()`            |

---

## Mini-Exercise

Build a mini CNN with:

* 2 Conv2D layers
* ReLU and MaxPooling after each
* Flatten + Dense to output 10 classes

1. Feed a dummy input of shape `[1, 3, 224, 224]` (PyTorch) or `[1, 224, 224, 3]` (TF)
2. Print the shape after each layer
3. Try replacing ReLU with LeakyReLU—observe differences

**Bonus:** Visualize the first convolutional layer filters (we’ll expand this in Chapter 17!)

---