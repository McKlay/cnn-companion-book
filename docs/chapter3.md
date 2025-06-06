---
hide:
  - toc
---

# Chapter 3: From Pixels to Model Input

> “*Your model is only as good as the input you feed it. Garbage in, garbage out—but beautifully preprocessed data in? That’s how deep learning begins.*”

---

## Why This Chapter Matters

At this point, you understand how images are stored and how to manipulate tensors. Now we take the next step: building a complete, robust input pipeline that takes an image from file system → tensor → model-ready format.

This chapter answers:

- How do you convert raw image data to a float32 tensor?

- What’s the difference between resizing and reshaping?

- Why do batch dimensions matter?

- What happens when you feed data into a Conv2D or Dense layer?

- How do PyTorch and TensorFlow differ in handling the image input flow?

Whether you're loading a single image for inference or setting up batches for training, this chapter will help you debug shape mismatches, clean up input pipelines, and feed data correctly into your network.

---

## Conceptual Breakdown

**🔹 Full Image Input Pipeline Overview**

Every image input to a CNN passes through a pipeline like this:
```css
File (JPEG/PNG)
 ↓
Load to memory (PIL / tf.io / OpenCV)
 ↓
Convert to RGB (if not already)
 ↓
Resize or reshape to match model expectations
 ↓
Convert to float32
 ↓
Normalize (0–1 or mean/std)
 ↓
Reorder dimensions if needed ([H, W, C] ↔ [C, H, W])
 ↓
Add batch dimension
 ↓
Feed to CNN layer
```

This process must be precise, especially when you're working with pretrained models or initializing new architectures.

---

**🔹 Resize vs Reshape**

Understanding this difference is critical.

- Resize changes the actual content dimensions (resampling, possibly distorting image slightly).

  - Example: resize 640×480 → 224×224

- Reshape changes the data layout without touching the content. Dangerous if shape is wrong!

  - Only use reshape if you're 100% sure of data layout.

*📌 Resizing is typically used for image preprocessing. Reshape is for tensor manipulation post-preprocessing.*

---

**🔹 Normalization: Why Float32 and 0–1?**

CNNs expect normalized input:

  - Pixel values from 0–255 are too large and make training unstable.

  - Convert to float32 (`/255.0`) or apply dataset-specific mean-std normalization.

Common norms:

  - `[0.0, 1.0]` scaling → generic models

  - `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` → ImageNet-pretrained models

---

**🔹 Batch Dimension: Don’t Forget!**

Even for *one* image, CNNs expect a batch:

  - Conv2D: Expects input shape `[N, C, H, W]` (PyTorch) or `[N, H, W, C]` (TF)

  - N is batch size: must be ≥1

  - Failing to add this leads to shape errors when feeding into models

*📌 Use `.unsqueeze(0)` (PyTorch) or `tf.expand_dims(..., axis=0)` (TensorFlow)*

---

**🔹 Feeding Into a Conv2D or Dense Layer**

CNNs process **4D tensors**:

  - PyTorch: `[batch, channels, height, width]`

  - TensorFlow: `[batch, height, width, channels]`

What happens internally:

  - Conv2D takes a window of pixels

  - Applies filters (kernels)

  - Outputs a feature map

  - The deeper the layers, the higher the abstraction

Dense layers flatten the features:

  - Input must be reshaped before connecting to `nn.Linear` or `Dense()`

  - Usually done with `.view(batch_size, -1)` or `tf.reshape(x, [batch_size, -1])`

  ---

## PyTorch Implementation

Here’s a full input-to-model example:
<details>
  <summary>View Pytorch implementation</summary>
    ```python
    from PIL import Image
    import torch
    import torchvision.transforms as T
    import torch.nn as nn

    # Load and preprocess
    image = Image.open("dog.png").convert("RGB")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),  # Converts [H,W,C] to [C,H,W] and scales to 0–1
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # Example model layer
    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
    output = conv(input_tensor)  # Output shape: [1, 16, 222, 222]
    ```

    If feeding into a dense layer later:
    ```python
    flattened = output.view(output.size(0), -1)  # Flatten to [batch, features]
    fc = nn.Linear(flattened.size(1), 10)
    logits = fc(flattened)
    ```
</details>

---

## TensorFlow Implementation

Same input pipeline in TensorFlow:
```python
import tensorflow as tf

# Load and preprocess
image = tf.io.read_file("dog.png")
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [224, 224])
image = tf.cast(image, tf.float32) / 255.0

# Normalize with ImageNet stats
mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])
image = (image - mean) / std

# Add batch dimension: [1, 224, 224, 3]
input_tensor = tf.expand_dims(image, axis=0)

# Conv2D example
conv = tf.keras.layers.Conv2D(16, 3)
output = conv(input_tensor)  # [1, 222, 222, 16]

# Flatten + Dense
flattened = tf.reshape(output, [1, -1])
dense = tf.keras.layers.Dense(10)
logits = dense(flattened)
```

---

## Framework Comparison Table

|Pipeline Step	      |PyTorch	                      |TensorFlow                                 |
|---------------------|-------------------------------|-------------------------------------------|
|Load image	          |Image.open().convert("RGB")	  |tf.io.read_file() + tf.image.decode_*()    |
|Resize	              |T.Resize((H, W))	              |tf.image.resize()                          |
|Convert to tensor	  |T.ToTensor()	                  |tf.cast(..., tf.float32) + divide          |
|Normalize	          |T.Normalize(mean, std)	        |Manual: (image - mean) / std               |
|Batch dimension	    |tensor.unsqueeze(0)	          |tf.expand_dims(tensor, axis=0)             |
|CNN input shape	    |[N, C, H, W]	                  |[N, H, W, C]                               |
|Flatten for Dense	  |.view(N, -1)	                  |tf.reshape(..., [N, -1])                   |

---

## Mini-Exercise

Objective: Build a complete image → tensor → CNN pipeline.

  - Choose any image and ensure it’s RGB.

  - Load and resize it to 224×224.

  - Normalize using ImageNet mean and std.

  - Add batch dimension and print final shape.

  - Feed into a Conv2D layer and flatten it.

  - Visualize the shape before and after each step.

Bonus Challenge:

  - Try with grayscale and handle single channel

  - Try with a batch of 5 images

---


