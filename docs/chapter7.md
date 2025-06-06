---
hide:
    - toc
---

# **Chapter 7: Data Augmentation Techniques (Expanded)**

> “*If your model memorizes your dataset, you’ve failed. Augmentation teaches it to imagine.*”

---

## **Why This Chapter Matters**

Most real-world datasets are:

   - Small

   - Biased

   - Repetitive

Without augmentation, your CNN learns to memorize patterns instead of generalizing. That’s why data augmentation is not just a “nice to have”—it’s a core strategy to help models perform better on unseen data.

In this chapter, we go beyond the basics:

   - You’ll learn classic augmentations like random crop, flip, and jitter

   - Then expand into modern techniques like Cutout, Mixup, and CutMix

   - And you’ll implement these in both PyTorch and TensorFlow, with visualization

---

## **Conceptual Breakdown**

**🔹 What is Data Augmentation?**

Augmentation is the process of applying random transformations to training images on the fly—so the model sees a new version of each image every epoch.

It’s only used during training, never during validation or inference.

---

**🔹 Classic Augmentations**

|Augmentation	      |Effect                                            |
|--------------------|--------------------------------------------------|
|RandomCrop	         |Focus on subregions, simulate framing variation   |
|HorizontalFlip	   |Simulate left-right symmetry                      |
|ColorJitter	      |Adjust brightness, contrast, saturation, hue      |
|Rotation	         |Handle orientation bias                           |
|Gaussian Blur	      |Simulate camera focus variation                   |

---

**🔹 Advanced Augmentations**

|Technique	|Description                                                         |
|-----------|--------------------------------------------------------------------|
|Cutout	   |Randomly removes a square region (forces model to focus elsewhere)  |
|Mixup	   |Blends two images and their labels linearly                         |
|CutMix	   |Combines patches from different images (and labels)                 |

---

**🔹 Why They Work**

   - Cutout teaches robustness to occlusion

   - Mixup teaches interpolation between classes

   - CutMix teaches spatial composition and label smoothing

> 📌 These augmentations improve generalization, reduce overfitting, and even improve model calibration.

---

## PyTorch Implementation

**🔸 Classic Augmentations**

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```

---

**🔸 Cutout (Custom)**

```python
import torch
import numpy as np
import torchvision.transforms.functional as F

class Cutout(object):
    def __init__(self, size=50):
        self.size = size

    def __call__(self, img):
        c, h, w = img.shape
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)

        img[:, y1:y2, x1:x2] = 0.0
        return img
```

Add it to your transform:

```python
train_transform.transforms.append(Cutout(size=32))
```

---

## **TensorFlow Implementation**

**🔸 Classic Augmentations**

```python
import tensorflow as tf
from tensorflow.keras import layers

data_augment = tf.keras.Sequential([
    layers.Resizing(256, 256),
    layers.RandomCrop(224, 224),
    layers.RandomFlip("horizontal"),
    layers.RandomBrightness(factor=0.2),
    layers.RandomContrast(factor=0.2),
])
```
Use during training:
```python
train_ds = train_ds.map(lambda x, y: (data_augment(x), y))
```

**🔸 CutMix**

```python
import tensorflow_addons as tfa

def cutmix(images, labels, alpha=1.0):
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    lam = tf.random.uniform([], 0, 1)
    image_shape = tf.shape(images)[1:]
    cut_w = tf.cast(image_shape[1] * tf.math.sqrt(1 - lam), tf.int32)
    cut_h = tf.cast(image_shape[0] * tf.math.sqrt(1 - lam), tf.int32)

    cx = tf.random.uniform([], 0, image_shape[1], dtype=tf.int32)
    cy = tf.random.uniform([], 0, image_shape[0], dtype=tf.int32)

    x1 = tf.clip_by_value(cx - cut_w // 2, 0, image_shape[1])
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, image_shape[0])
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, image_shape[1])
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, image_shape[0])

    padding = [[0, 0], [y1, image_shape[0] - y2], [x1, image_shape[1] - x2], [0, 0]]
    cutmix_img = tf.pad(shuffled_images, padding, constant_values=0)

    new_images = tf.tensor_scatter_nd_update(images, [[0]], [cutmix_img])
    new_labels = lam * labels + (1 - lam) * shuffled_labels

    return new_images, new_labels
```

> 📌 You can also use TensorFlow Addons or Albumentations for more advanced pipelines.

---

## **Framework Comparison Table**

|Augmentation	      |PyTorch (torchvision)	               |TensorFlow (Keras or tf.data)            |
|--------------------|--------------------------------------|-----------------------------------------|
|Resize/Crop/Flip	   |`transforms.*`	                     |`layers.Resizing()`, `layers.Random*()`  |
|Color Jitter	      |`transforms.ColorJitter()`	         |`layers.RandomBrightness()`, etc.        |
|Cutout	            |Custom class	                        |Custom or `tfa.image.random_cutout()`    |
|Mixup	            |Custom function	                     |Custom function or `tf.image` logic      |
|CutMix	            |Custom function	                     |TensorFlow Addons or custom logic        |
|Batch-safe usage	   |`transforms.Compose()` + DataLoader	|`.map(lambda x, y: ...)` in `tf.data`    |

---

## M**ini-Exercise**

1. Pick a sample dataset (e.g., 100 dog and cat images)

2. Apply:

&nbsp;&nbsp;&nbsp;&nbsp; 🔸 RandomCrop + Flip + ColorJitter (PyTorch)

&nbsp;&nbsp;&nbsp;&nbsp; 🔸 Resize + RandomBrightness (TF)

3. Implement:

&nbsp;&nbsp;&nbsp;&nbsp; 🔸 Cutout in PyTorch

&nbsp;&nbsp;&nbsp;&nbsp; 🔸 Mixup or CutMix in TensorFlow

4. Visualize 5 examples before and after augmentation

5. Train a simple CNN with and without augmentation—observe accuracy

---





