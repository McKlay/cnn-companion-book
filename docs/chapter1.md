---
hide:
  - toc
---

# Chapter 1: How a Neural Network Sees an Image

> “*Before the model learns, it sees. Before it classifies, it computes. And what it sees—starts with pixels, channels, and shapes.*”

---

## Why This Chapter Matters

Every computer vision journey begins with an image. But here’s the twist: your neural network doesn’t see an image the way you do. It sees numbers. And not just any numbers—tensors of pixel values, reshaped and normalized to fit the model’s expectations.

If you’ve ever run into errors like:

- “Expected 3 channels, got 1”

- “Shape mismatch: [1, 224, 224, 3] vs [3, 224, 224]”

- “Model output is garbage despite clean code”

…then it probably started here: the image-to-tensor pipeline wasn’t correctly handled.

In this chapter, we’ll unpack the complete transformation from a JPEG or PNG file on disk to a model-ready tensor in memory. We’ll go step by step—from pixel arrays → float tensors → properly shaped inputs—and explain how frameworks like PyTorch and TensorFlow treat the process differently.

You’ll see what the model sees. And that understanding will anchor everything you build later.

---

## Conceptual Breakdown

**🔹 What Is an Image in Memory?**

To a neural network, an image is just a 3D array—Height, Width, and Color Channels (usually RGB). For grayscale, it’s just H×W. For RGB, it’s H×W×3.

But raw image files (JPEG, PNG) are compressed formats. To use them in training, we:

1. **Load** the image into memory

2. **Convert** it to an array of pixel values (0–255)

3. **Normalize/scale** those values (e.g., 0.0 to 1.0 or with mean/std)

4. **Reshape** it into a tensor format the model expects

Each step matters. A mismatch in any of these can wreck your model.

---

**🔹 Tensor Layouts: [H, W, C] vs [C, H, W]**

Different frameworks use different conventions:

- TensorFlow uses `[Height, Width, Channels]`

- PyTorch uses `[Channels, Height, Width]`

The reason? Internal memory layout optimizations. But for you, it means that converting between these shapes is crucial when preparing images for your models.

---

**🔹 Model Input Shape: Why It Matters**

Neural networks are strict about input shape:

- ResNet, MobileNet, EfficientNet, etc. expect a specific input size and layout

- Channels must match: grayscale (1), RGB (3), etc.

- Batch dimension must exist: `[1, C, H, W]` or `[1, H, W, C]`

Even for a single image, you **must simulate a batch**—most models don’t accept raw 3D tensors.

---

**🔹 Visual Walkthrough: Image → Tensor → Model**

Let’s break down what happens:
```css
Image File (e.g., 'dog.png')
      ↓
Load into memory (PIL / tf.io / OpenCV)
      ↓
Convert to NumPy or Tensor (shape: H×W×3)
      ↓
Normalize (e.g., /255.0 or mean/std)
      ↓
Transpose (if using PyTorch: → C×H×W)
      ↓
Add batch dim (→ 1×C×H×W or 1×H×W×C)
      ↓
Feed to CNN
```

---

## PyTorch Implementation

Here’s how you go from image file to model-ready tensor in PyTorch:
```python
from PIL import Image
import torchvision.transforms as T

# 1. Load image
image = Image.open("dog.png").convert("RGB")

# 2. Define transform pipeline
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),  # Converts to [0,1] and switches to [C, H, W]
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])  # Pretrained model mean/std
])

# 3. Apply transforms
tensor = transform(image)  # shape: [3, 224, 224]

# 4. Add batch dimension
input_tensor = tensor.unsqueeze(0)  # shape: [1, 3, 224, 224]
```

---

## TensorFlow Implementation

The same pipeline in TensorFlow looks like this:
```python
import tensorflow as tf

# 1. Load image
image = tf.io.read_file("dog.png")
image = tf.image.decode_png(image, channels=3)

# 2. Resize and convert to float32
image = tf.image.resize(image, [224, 224])
image = tf.cast(image, tf.float32) / 255.0

# 3. Normalize
mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])
image = (image - mean) / std

# 4. Add batch dimension
input_tensor = tf.expand_dims(image, axis=0)  # shape: [1, 224, 224, 3]
```

---

# Framework Comparison Table
|Step	            |PyTorch	                        |TensorFlow                             |
|-------------------|-----------------------------------|---------------------------------------|
|Load image	        |`PIL.Image.open()`	                |`tf.io.read_file() + tf.image.decode`  |
|Resize	            |`T.Resize((H, W))`	                |`tf.image.resize()`                    |
|Convert to float	|`T.ToTensor()` (scales to 0–1)	    |`tf.cast(..., tf.float32) / 255.0`     |
|Normalize	        |`T.Normalize(mean, std)`	        |Manual: `(image - mean) / std`         |
|Layout	            |`[C, H, W]`	                    |`[H, W, C]`                            |
|Add batch dim	    |`.unsqueeze(0)`	                |`tf.expand_dims(..., axis=0)`          |

---

## Mini-Exercise

Choose any image file and:

1. Load and visualize the original

2. Convert it to a tensor using both PyTorch and TensorFlow

3. Apply normalization

4. Print shape at each step

5. Confirm final shape matches model input requirement

**Bonus**: Try visualizing the image after normalization. What do the pixel values look like now?

---




