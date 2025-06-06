---
hide:
    - toc
---

# Chapter 9: *The CNN Vocabulary (Terms Demystified)*

> *“Before you build deep networks, build deep understanding. Words like kernel, stride, and feature map aren’t just jargon—they’re the gears of a vision engine.”*

---

## Why This Chapter Matters

If you’ve ever wondered:

* “What exactly is a kernel?”
* “How do channels differ from filters?”
* “Why does stride affect output shape?”
* “What’s the difference between padding types?”

… then this chapter is for you.

Clear understanding of these terms helps you:

* **Design architectures confidently**
* **Avoid shape mismatch bugs**
* **Communicate ideas and debug issues quickly**
* **Understand pretrained model behavior**

---

## Conceptual Breakdown

Let’s define and **visually ground** each essential CNN term.

---

### 🔹 Kernel (a.k.a. Filter)

**What it is:** A small matrix (e.g., 3×3 or 5×5) that slides across the image, performing local dot products.

* Each kernel learns to detect a pattern (e.g., edge, curve, texture)
* A Conv2D layer contains **many kernels**—one per output channel

Think of a kernel as the "eye" scanning a small area.

| Size | Meaning                  |
| ---- | ------------------------ |
| 1×1  | Channel-wise projection  |
| 3×3  | Local feature extraction |
| 5×5  | More context, costlier   |

---

### 🔹 Stride

**What it is:** The number of pixels the kernel moves each time.

* Stride = 1 → overlapping windows
* Stride = 2 → skips every other pixel, downsamples output

Stride controls **spatial resolution** of the output.

---

### 🔹 Padding

**What it is:** How we handle the edges of the image.

| Type   | Description                                      |
| ------ | ------------------------------------------------ |
| Valid  | No padding (output shrinks)                      |
| Same   | Pads so output shape matches input (if stride=1) |
| Custom | Manually pad with specific values                |

📌 In PyTorch: `padding=1` for 3×3 kernel maintains shape
📌 In TensorFlow: use `padding='same'` or `'valid'`

---

### 🔹 Input/Output Channels

**Input Channels:** Number of channels in the incoming tensor
**Output Channels:** Number of filters (each outputs a channel)

| Layer  | Input Shape          | Output Shape                 |
| ------ | -------------------- | ---------------------------- |
| Conv2D | `[B, 3, H, W]` (RGB) | `[B, 64, H, W]` (64 filters) |

Every output channel corresponds to one kernel applied across all input channels.

---

### 🔹 Feature Maps

**What it is:** The output of a convolution layer—a 2D activation map showing how strongly a feature was detected in different regions.

* Early layers: feature maps detect **edges, corners**
* Deeper layers: feature maps detect **eyes, wheels, textures**

📌 Feature maps = filtered views of the image.

---

### 🔹 Receptive Field

**What it is:** The **effective area** of the original input that a neuron “sees.”

* Grows with depth
* A neuron in a deep layer might “see” the entire image

A large receptive field = global understanding
Small receptive field = local detail

---

### 🔹 Channel Depth vs Spatial Dimensions

| Property     | Meaning                     |
| ------------ | --------------------------- |
| Spatial size | Height × Width (resolution) |
| Depth        | Number of feature channels  |

Example: `[32, 128, 128]` = 32 filters, 128×128 resolution per map

---

### 🔹 Layer Variants

| Term                  | Meaning                                                                 |
| --------------------- | ----------------------------------------------------------------------- |
| **ReflectionPad2d**   | Pads by mirroring the image at the edge (used in style transfer)        |
| **InstanceNorm2d**    | Like BatchNorm, but per image-instance (used in image generation tasks) |
| **AdaptiveAvgPool2d** | Automatically resizes output to fixed size regardless of input size     |

These are powerful tools when building **style-transfer**, **GANs**, or **segmentation** models.

---

## PyTorch Examples

```python
import torch.nn as nn

# 3x3 conv, same output shape
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

# Pooling
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Adaptive pooling to 1×1 (useful before a Dense layer)
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

# Reflection padding (e.g., style transfer)
pad = nn.ReflectionPad2d(2)

# Instance normalization (used in generator networks)
norm = nn.InstanceNorm2d(16)
```

---

## TensorFlow Examples

```python
from tensorflow.keras import layers

# Conv with SAME padding
conv = layers.Conv2D(16, kernel_size=3, padding='same')

# Max Pooling
pool = layers.MaxPooling2D(pool_size=(2, 2), strides=2)

# Adaptive pooling (Global Average Pool)
adaptive = layers.GlobalAveragePooling2D()

# Reflection padding: must be done manually
padded = tf.pad(input_tensor, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='REFLECT')

# Instance norm (use tf_addons or custom layer)
```

---

## Framework Comparison Table

| Concept            | PyTorch                          | TensorFlow                           |
| ------------------ | -------------------------------- | ------------------------------------ |
| Conv2D             | `nn.Conv2d(in, out, k)`          | `layers.Conv2D(out, k, padding=...)` |
| Padding (same)     | `padding=1` (for 3×3)            | `padding='same'`                     |
| Adaptive pooling   | `AdaptiveAvgPool2d(output_size)` | `GlobalAveragePooling2D()`           |
| InstanceNorm       | `nn.InstanceNorm2d()`            | Addons/custom implementation         |
| Reflection padding | `nn.ReflectionPad2d(pad)`        | `tf.pad(..., mode='REFLECT')`        |

---

## Mini-Exercise

Choose an image and:

1. Manually implement:

   * A 3×3 Conv2D with stride 1 and padding 1
   * A MaxPool2D with stride 2
   * A GlobalAveragePooling layer

2. Print the shape of each output step-by-step

3. Visualize:

   * The input
   * The output feature maps of the first convolution

**Bonus**: Try using `AdaptiveAvgPool2d((1, 1))` to make your model input-shape agnostic.

---

