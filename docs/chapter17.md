---
hide:
    - toc
---

Wonderful, Clay. Let’s now uncover what CNNs are really “seeing” inside those hidden layers.

This chapter makes the invisible—visible. When your CNN makes a decision, it's because it “sees something” in the input image. But *what* is it seeing? That’s where **feature map visualization** comes in.

---

# Chapter 17: *Visualizing Feature Maps and Filters*

> *“CNNs are not black boxes. You just need the right lens to peek inside.”*

---

## Why This Chapter Matters

As your models become more complex, the need for interpretability becomes critical:

* Are filters detecting edges, textures, or irrelevant noise?
* Why did my CNN misclassify an image?
* Is it paying attention to the **right part** of the image?

**Feature visualization** answers these by revealing:

* **What each layer activates on**
* **How filters evolve with training**
* **Whether attention focuses on objects or noise**

This chapter teaches:

* How to extract **intermediate outputs** (feature maps)
* Visualize **filters** learned by convolutional layers
* Build **debugging tools** for visual interpretation

---

## Conceptual Breakdown

### 🔹 What Are Feature Maps?

A **feature map** is the output of a convolution layer:

* It shows **activation patterns** for a given image
* Early maps detect **edges, textures**
* Deeper maps focus on **shapes, semantics**

A single input image generates **many feature maps**, one for each filter (channel) in the layer.

---

### 🔹 What Are Filters?

Filters are the learnable **weight matrices** applied during convolution:

* Size: typically 3×3 or 5×5
* Shape: `[out_channels, in_channels, kernel_size, kernel_size]`

Visualizing filters helps you:

* Understand learned patterns
* Identify dead or redundant filters
* Compare pretrained vs randomly initialized filters

---

## PyTorch Implementation

### 🔸 1. Visualizing Intermediate Feature Maps

Let’s visualize the output after the first convolutional block.

#### Step 1: Register Forward Hook

```python
activations = {}

def hook_fn(module, input, output):
    activations['conv1'] = output.detach()

model.conv1.register_forward_hook(hook_fn)
```

#### Step 2: Forward Pass an Image

```python
model.eval()
with torch.no_grad():
    _ = model(input_image.unsqueeze(0))  # Batch of 1
```

#### Step 3: Visualize the Activation Maps

```python
import matplotlib.pyplot as plt

act = activations['conv1'].squeeze()  # shape: [num_filters, H, W]

fig, axes = plt.subplots(4, 8, figsize=(12, 6))  # for 32 filters
for i, ax in enumerate(axes.flat):
    ax.imshow(act[i].cpu(), cmap='viridis')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

### 🔸 2. Visualizing Filters

```python
filters = model.conv1.weight.data.clone()  # shape: [out_channels, in_channels, k, k]

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    filt = filters[i, 0]  # visualize first channel of each filter
    ax.imshow(filt.cpu(), cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

## TensorFlow Implementation

### 🔸 1. Define Sub-Model for Intermediate Layers

Keras makes this straightforward:

```python
from tensorflow.keras.models import Model

layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
```

### 🔸 2. Forward Pass and Visualize

```python
activations = activation_model.predict(input_image[None, ...])  # Add batch dim

first_layer_activations = activations[0]  # shape: [1, H, W, filters]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(first_layer_activations[0, :, :, i], cmap='viridis')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

### 🔸 3. Visualize Filters of a Conv Layer

```python
weights = model.get_layer('conv2d').get_weights()[0]  # shape: (k, k, in_channels, out_channels)

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    filt = weights[:, :, 0, i]  # visualizing first input channel
    ax.imshow(filt, cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

## Framework Comparison Table

| Task                     | PyTorch                    | TensorFlow/Keras                                |
| ------------------------ | -------------------------- | ----------------------------------------------- |
| Extract feature map      | Use forward hook           | Use `Model(inputs, outputs)`                    |
| Visualize filter weights | Access `layer.weight.data` | `layer.get_weights()`                           |
| View multiple layers     | Register multiple hooks    | Output list from sub-model                      |
| Forward inference        | `with torch.no_grad()`     | `model.predict()` or `model(x, training=False)` |
| Activation shape         | `[batch, channels, H, W]`  | `[batch, H, W, channels]`                       |

---

## Mini-Exercise

1. Choose any pretrained model (ResNet18, MobileNet, etc.)
2. Input a single image of a dog or cat
3. Extract and visualize:

   * Filters from the first Conv2D layer
   * Activation maps from the first 2 convolutional blocks
4. Interpret:

   * Which filters activate strongest?
   * What kind of patterns do they seem to detect?
5. Bonus:

   * Try with a different image (car, flower) and compare changes

---

## Debugging Insights

Visualizations help you discover:

* **Dead filters**: filters with near-zero activation across samples
* **Bias**: model focusing on background rather than subject
* **Overfitting**: filters overly tuned to irrelevant details

You can also visualize *wrong predictions* to understand **why** your model failed.

---

## What You Can Now Do

* Open the black box of your CNN models
* Debug incorrect outputs by checking what the model “sees”
* Inspect learned filters to ensure diversity and relevance
* Use visualization as a sanity check during training

---