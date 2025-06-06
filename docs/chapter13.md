---
hide:
    - toc
---

# Chapter 13: *Loss Functions and Optimizers*

> *“A network doesn’t improve by magic—it learns by failing. The loss is the pain, the optimizer is the cure.”*

---

## Why This Chapter Matters

When training a CNN, we want it to:

* Make better predictions over time
* Improve by adjusting its weights through **gradients**

But **how do we quantify wrong predictions?**
That’s where **loss functions** come in.

And once we have loss, **how do we adjust the network to reduce it?**
That’s the job of **optimizers**.

Together, they are the **learning engine** of your CNN:

* The **loss function** tells the model how wrong it is.
* The **optimizer** updates the weights to make it less wrong next time.

Understanding both is vital for:

* Choosing the right learning strategy
* Debugging model collapse or instability
* Fine-tuning pretrained models

---

## Conceptual Breakdown

### 🔹 What Is a Loss Function?

A **loss function** computes the difference between:

* The model’s predicted output (logits/probabilities)
* The true label (target)

The output is a **scalar**, which is **differentiable** so gradients can flow.

### 🔹 Types of Loss Functions

| Loss Function         | Use Case                            | PyTorch                  | TensorFlow                             |
| --------------------- | ----------------------------------- | ------------------------ | -------------------------------------- |
| **CrossEntropyLoss**  | Multi-class classification          | `nn.CrossEntropyLoss()`  | `SparseCategoricalCrossentropy()`      |
| **BCEWithLogitsLoss** | Binary classification (with logits) | `nn.BCEWithLogitsLoss()` | `BinaryCrossentropy(from_logits=True)` |
| **MSELoss**           | Regression or feature matching      | `nn.MSELoss()`           | `MeanSquaredError()`                   |

📌 For classification tasks:

* Use **CrossEntropyLoss** if your model outputs raw **logits**
* For softmax outputs, use **CategoricalCrossentropy** without logits

---

### 🔹 What Is an Optimizer?

An **optimizer** updates weights using the gradients computed via backpropagation.

Optimizers apply:

* Learning rate (`lr`)
* Momentum, adaptive steps, or regularization

---

### 🔹 Common Optimizers

| Optimizer          | Behavior                         | Best For                              |
| ------------------ | -------------------------------- | ------------------------------------- |
| **SGD**            | Basic gradient descent           | Simple, interpretable tasks           |
| **SGD + Momentum** | Adds velocity to updates         | Faster convergence                    |
| **Adam**           | Adaptive step size per parameter | Most deep learning tasks              |
| **RMSprop**        | Like Adam but simpler            | Good for noisy gradients (e.g., RNNs) |

📌 **Start with Adam.** Move to SGD + Momentum for fine-tuning large models.

---

### 🔹 Visualizing Gradient Flow

Think of:

* **Loss** as elevation
* **Gradients** as slope
* **Optimizer** as the hiker moving downhill

Bad loss or bad optimizer = stuck in a valley
Good setup = smooth descent to a better model

---

## PyTorch Implementation

### 🔸 CrossEntropy Loss + Adam Optimizer

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = MiniCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training step
for images, labels in train_loader:
    optimizer.zero_grad()        # Reset gradients
    outputs = model(images)      # Forward pass
    loss = criterion(outputs, labels)  # Compute loss
    loss.backward()              # Backpropagation
    optimizer.step()             # Update weights
```

### 🔸 Switch to SGD with Momentum

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

## TensorFlow Implementation

### 🔸 CrossEntropy Loss + Adam Optimizer

```python
import tensorflow as tf

model = MiniCNN()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Dummy training step (eager)
with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_fn(labels, predictions)
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

### 🔸 Use SGD Instead

```python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
```

---

## Framework Comparison Table

| Component        | PyTorch                                | TensorFlow                                        |
| ---------------- | -------------------------------------- | ------------------------------------------------- |
| Loss Function    | `nn.CrossEntropyLoss()`                | `tf.keras.losses.SparseCategoricalCrossentropy()` |
| Loss with logits | Built-in (raw outputs supported)       | `from_logits=True`                                |
| Optimizer        | `optim.Adam(...)`                      | `tf.keras.optimizers.Adam(...)`                   |
| Update weights   | `loss.backward()` + `optimizer.step()` | `GradientTape()` + `apply_gradients()`            |
| Zero gradients   | `optimizer.zero_grad()`                | Automatic inside tape                             |

---

## Mini-Exercise

1. Build a CNN for 10-class image classification
2. Try both **CrossEntropy + Adam** and **SGD + momentum**
3. Log loss per batch and plot curve after 5 epochs
4. Try tweaking:

   * Learning rate (`lr`)
   * Loss function (`BCEWithLogitsLoss` for binary)
5. Observe: Does one optimizer converge faster? Does one oscillate?

**Bonus**: Add weight decay (L2 regularization) to SGD and test performance.

---