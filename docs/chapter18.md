---
hide:
    - toc
---

# Chapter 18: *Inference Pipeline Design*

> *“Training wins the accuracy race. Inference wins the deployment game.”*

---

## Why This Chapter Matters

No matter how well your model performs in training, it’s useless if it fails at inference time.

Here’s where things often go wrong:

* You trained with `[0,1]` scaled images but used `[-1,1]` scaling at inference
* You forgot to switch to `.eval()` mode or `training=False`
* You tested with images of a different size than during training
* Your real-world images have noise, padding, or background not present in your dataset

**Inference is a system, not just a `.predict()` call.**

This chapter shows you how to:

* Design **reusable**, **consistent**, and **fault-tolerant** pipelines
* Align preprocessing between **training and deployment**
* Apply **test-time augmentation (TTA)** for better accuracy
* Defend your model against bad input or unexpected formats

---

## Conceptual Breakdown

### 🔹 What is an Inference Pipeline?

It’s the **entire path** an image takes from user input to model prediction:

1. Image is uploaded, captured, or streamed
2. Preprocessing (resize, normalize, etc.)
3. Passed through model (in eval mode, no gradients)
4. Output is decoded (softmax, argmax, etc.)
5. Results are returned in user-friendly format

> A mistake at **any** step will lead to wrong predictions.

---

### 🔹 Training vs Inference: Matching Pipelines

| Stage         | During Training                              | During Inference            |
| ------------- | -------------------------------------------- | --------------------------- |
| Resize        | `Resize((224, 224))`                         | Same exact shape required   |
| Normalization | `Normalize(mean, std)` or `rescale to [0,1]` | Must match exactly          |
| Augmentations | RandomCrop, Flip, ColorJitter (for variety)  | Disabled, or TTA only       |
| Mode          | `model.train()`                              | `model.eval()`              |
| Gradients     | `requires_grad=True`                         | `no_grad()` / tape disabled |

If you change any of the above during inference, your model may misbehave.

---

## PyTorch Implementation

### 🔸 1. Reusable Preprocessing Function

```python
from torchvision import transforms

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Must match training!
    ])
```

### 🔸 2. Inference Function

```python
from PIL import Image
import torch

def predict_image(model, image_path, transform):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()
    return prediction
```

---

## TensorFlow Implementation

### 🔸 1. Preprocessing Function

For models like MobileNet or EfficientNet, use built-in preprocessors:

```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

def prepare_image_tf(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # Handles [-1,1] scaling
    return np.expand_dims(img_array, axis=0)
```

### 🔸 2. Inference Function

```python
def predict_tf(model, img_path):
    img_tensor = prepare_image_tf(img_path)
    predictions = model.predict(img_tensor)
    return np.argmax(predictions)
```

---

## Additions for a Full Inference System

| Component                  | Why It Matters                          |
| -------------------------- | --------------------------------------- |
| **Input validation**       | Ensure correct shape, color channels    |
| **Test-Time Augmentation** | Improve prediction by averaging outputs |
| **Softmax thresholding**   | Avoid low-confidence predictions        |
| **Postprocessing**         | Map label index → human-readable class  |
| **Batching**               | Speed up inference for multiple inputs  |

---

### 🔸 Optional: Test-Time Augmentation (TTA)

Run multiple variants of the same image and average predictions.

```python
def tta_predict(model, image, transforms_list):
    outputs = []
    for t in transforms_list:
        img = t(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img)
        outputs.append(output)
    return torch.stack(outputs).mean(dim=0).argmax().item()
```

---

## Framework Comparison Table

| Feature                   | PyTorch                            | TensorFlow / Keras                            |
| ------------------------- | ---------------------------------- | --------------------------------------------- |
| Eval mode                 | `model.eval()`                     | `training=False` in `model(x, training=...)`  |
| Gradient-free inference   | `with torch.no_grad()`             | Default in `model.predict()`                  |
| Reusable preprocessing    | `torchvision.transforms.Compose()` | `keras.preprocessing` or `tf.image`           |
| Built-in TTA              | Manual                             | Manual                                        |
| Model saving              | `torch.save(model.state_dict())`   | `model.save()` to SavedModel format           |
| Normalization consistency | User-defined                       | Use `keras.applications.*.preprocess_input()` |

---

## Mini-Exercise

Build a full inference function that:

1. Accepts an image path
2. Applies identical preprocessing from training
3. Loads a trained model
4. Switches to inference mode
5. Predicts the class and returns a human-readable label

**Bonus**:

* Add test-time augmentation
* Log input/output shape and prediction confidence

---

## Gotchas to Watch Out For

| Problem                           | Likely Cause                                |
| --------------------------------- | ------------------------------------------- |
| Model always predicts same class  | Forgetting `.eval()` or bad normalization   |
| High training accuracy, poor test | Mismatched preprocessing (e.g., RGB to BGR) |
| Inference crashes on large input  | Missing batch dimension or wrong shape      |
| Weird predictions at deployment   | Dropout still active, or inconsistent mode  |

---

## What You Can Now Do

* Write a robust inference script from scratch
* Detect input shape and channel mismatches
* Reuse training transforms to guarantee consistency
* Use test-time augmentation to improve generalization
* Ship CNNs in reproducible, traceable pipelines

---