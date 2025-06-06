---
hide:
    - toc
---

# 📘 Chapter 10: *Writing the `forward()` / `call()` Function*

> *“This is where it all flows. The forward pass isn’t just where your model computes—it’s where it thinks.”*

---

## Why This Chapter Matters

Designing a CNN is one thing—**executing it** is another. All the layers you define? They don’t mean anything until you stitch them together in a **forward flow of logic.**

This chapter teaches you how to:

* Build **custom models** using `nn.Module` and `tf.keras.Model`
* Control **data flow** through convolutional, pooling, and linear layers
* Debug shape mismatches through **layer-by-layer tracking**
* Make your model clean, modular, and ready for real-world deployment

When implemented right, your model becomes not just correct—but **elegant and debuggable**.

---

## Conceptual Breakdown

### 🔹 What Is the `forward()` / `call()` Function?

This function defines **how your model processes data.** It tells the model:

* What layers to use
* In what order to apply them
* What transformations to perform

📌 *Think of this function as the “neural circuit diagram.”*

---

### 🔹 PyTorch: `forward()`

* Defined inside a subclass of `nn.Module`
* Called **automatically** during training or inference

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16 * 222 * 222, 10)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

---

### 🔹 TensorFlow: `call()`

* Defined inside a subclass of `tf.keras.Model`
* Called implicitly during training (`fit`) or explicitly with `model(x)`

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(16, 3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.conv(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        return self.fc(x)
```

---

### 🔹 Best Practices

| Tip                                                                    | Why It Matters                         |
| ---------------------------------------------------------------------- | -------------------------------------- |
| Keep your model class clean                                            | Separates definition from execution    |
| Use `nn.Sequential` or functional blocks                               | Reuse logic and reduce clutter         |
| Print shapes in `forward()`                                            | Helps catch shape mismatches early     |
| Use `x.view(x.size(0), -1)` (PyTorch) or `Flatten()` (TF) before Dense | Proper vector flattening for FC layers |

---

## PyTorch Implementation

Let’s create a full CNN with modular, readable `forward()` logic.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),  # OR: use .view() in forward
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        print("Input:", x.shape)
        x = self.features(x)
        print("After Conv Layers:", x.shape)
        x = self.classifier(x)
        print("Output:", x.shape)
        return x
```

📌 This model assumes input shape `[B, 3, 224, 224]`. You can adjust the classifier input size if using a different input resolution.

---

## TensorFlow Implementation

Now the same logic in TensorFlow using `tf.keras.Model` subclassing:

```python
import tensorflow as tf

class ConvClassifier(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(2)

        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(2)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, x, training=False):
        print("Input:", x.shape)
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        print("After Conv Layers:", x.shape)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)
```

---

## Debugging Tip: Shape-by-Shape Logging

To debug model flow:

* In **PyTorch**, use `print(x.shape)` inside `forward()`
* In **TensorFlow**, use `print(x.shape)` inside `call()`

---

## Framework Comparison Table

| Task                    | PyTorch                                     | TensorFlow                             |
| ----------------------- | ------------------------------------------- | -------------------------------------- |
| Model base class        | `nn.Module`                                 | `tf.keras.Model`                       |
| Define layers           | In `__init__()`                             | In `__init__()`                        |
| Execute layers          | In `forward(self, x)`                       | In `call(self, x)`                     |
| Activation example      | `F.relu(x)` or `nn.ReLU()`                  | `tf.nn.relu(x)` or `activation='relu'` |
| Flatten                 | `x.view(x.size(0), -1)` or `nn.Flatten()`   | `Flatten()` layer                      |
| Dropout (training only) | `nn.Dropout(p)` (active only in `.train()`) | `Dropout()(x, training=training)`      |

---

## Mini-Exercise

Create a small CNN model using both frameworks that:

1. Accepts input shape `[1, 3, 64, 64]` (PyTorch) or `[1, 64, 64, 3]` (TF)
2. Contains:

   * Two `Conv2D` layers
   * `ReLU` + `MaxPool` after each
   * One `Flatten` + `Dense` + `Dropout`
3. Prints shape at each stage of `forward()` or `call()`
4. Returns logits for 5 classes

**Bonus**: Add a conditional block to test `training=True` for dropout behavior in TensorFlow.

---

