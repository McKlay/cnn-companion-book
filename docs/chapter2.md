---
hide:
  - toc
---

# Chapter 1: What is a Tensor (in Code and in Mind)?

> “*You can’t debug what you don’t understand. And in deep learning, most confusion begins with shapes.*”

---

## Why This Chapter Matters

Before we train any CNN, before we pass an image through any layer, there’s something more foundational: the tensor. It’s not just a data structure—it’s the very language your model thinks in.

You might know it as just a NumPy array or a multi-dimensional grid. But in computer vision, a tensor carries the shape of your reality. Understanding how to think in tensors is what separates beginners who get stuck at shape mismatches… from engineers who can flow seamlessly between [C, H, W] and [H, W, C], across any framework.

In this chapter, we’ll demystify tensors—not just how to use them in PyTorch and TensorFlow, but how to think about them in your head. We’ll learn to reshape, permute, slice, and batch like second nature.

---

## Conceptual Breakdown

### 🔹 What is a tensor?

A tensor is a generalization of scalars, vectors, and matrices into higher dimensions.

- A scalar is a 0D tensor: 5

- A vector is a 1D tensor: [5, 3, 2]

- A matrix is a 2D tensor: [[1, 2], [3, 4]]

- An image is typically a 3D tensor: Channels × Height × Width or Height × Width × Channels

- A batch of images? A 4D tensor.

So, for example:
```text
A grayscale image (28x28) → [28, 28]  
An RGB image (224x224) → [3, 224, 224] or [224, 224, 3]  
A batch of RGB images (32 images) → [32, 3, 224, 224] or [32, 224, 224, 3]
```
Tensors store both the data and the shape. Understanding and controlling that shape is critical.

---

## PyTorch Implementation
In PyTorch, tensors are created and manipulated using `torch.tensor`, and reshaped with `.view()`, .`reshape()`, and `.permute()`.

**🔸 Basic Creation**

```python
import torch

scalar = torch.tensor(3)  # 0D
vector = torch.tensor([1, 2, 3])  # 1D
matrix = torch.tensor([[1, 2], [3, 4]])  # 2D
image = torch.rand(3, 224, 224)  # 3D image tensor (C, H, W)
```

---

**🔸 Reshaping and Permuting**

```python
# Reshape: flatten a tensor
image_flat = image.view(-1)  # Total elements = 3*224*224

# Permute: switch dimensions
image_hw_c = image.permute(1, 2, 0)  # Now (H, W, C)

# Add batch dimension
image_batch = image.unsqueeze(0)  # (1, 3, 224, 224)
```

---

## TensorFlow Implementation
In TensorFlow, we use `tf.constant`, `tf.reshape()`, and `tf.transpose()`.

**🔸 Basic Creation**
```python
import tensorflow as tf

scalar = tf.constant(3)  # 0D
vector = tf.constant([1, 2, 3])  # 1D
matrix = tf.constant([[1, 2], [3, 4]])  # 2D
image = tf.random.uniform((224, 224, 3))  # (H, W, C)
```

---

**🔸 Reshaping and Transposing**

```python
# Flatten
image_flat = tf.reshape(image, [-1])

# Transpose: (H, W, C) → (C, H, W)
image_chw = tf.transpose(image, [2, 0, 1])

# Add batch dimension
image_batch = tf.expand_dims(image, axis=0)  # (1, 224, 224, 3)
```

---

## Framework Comparison Table

|Concept	              |PyTorch	              |TensorFlow                        |
|---------------------------|---------------------------|----------------------------------|
|Tensor class	              |torch.Tensor	              |tf.Tensor                         |
|Shape format (image)	|[C, H, W]	              |[H, W, C]                         |
|Reshape	              |.view(), .reshape()	       |tf.reshape()                      |
|Transpose / Permute	       |.permute(dim_order)	       |tf.transpose(dim_order)           |
|Add batch dimension	       |.unsqueeze(dim)	       |tf.expand_dims(tensor, axis)      |
|Flatten	              |.view(-1)	              |tf.reshape(tensor, [-1])          |

---

## Mini-Exercise

Try loading a single image and perform the following in both PyTorch and TensorFlow:

1. Load the image as an array (e.g., with PIL or `tf.io`)

2. Convert it to a tensor

3. Normalize the pixel values to `[0, 1]`

4. Add a batch dimension

5. Ensure the shape is correct for your framework (PyTorch: `[1, 3, H, W]`, TF: `[1, H, W, 3]`)

---
