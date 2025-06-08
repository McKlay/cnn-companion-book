---
hide:
    - toc
---

# Chapter 14: *Training Loop Mechanics*

> *“This is where the magic happens—not in the layers, not in the loss—but in the loop where learning actually unfolds.”*

---

## Why This Chapter Matters

A CNN’s training process is a loop—a cycle that feeds data into the model, computes the loss, updates weights, and repeats across epochs. But training isn’t just calling `.fit()` or `.train()` and walking away.

You need to:

* **Log losses** and accuracy properly
* **Save and restore checkpoints**
* **Debug silently failing models**
* Use early stopping, learning rate schedules, and more

This chapter gives you the tools to:

* Write custom, reproducible training loops
* Understand what happens at every step
* Monitor model progress and troubleshoot problems early

---

## Conceptual Breakdown

### 🔹 Anatomy of a Training Loop

A complete training loop typically includes:

1. **Model in train mode**
2. Loop over **epochs**
3. Loop over **batches**
4. **Forward pass** through model
5. **Compute loss**
6. **Backward pass** (PyTorch) or **gradient tape** (TF)
7. **Update weights**
8. Track and log metrics
9. Validate model at each epoch

---

### 🔹 Epoch vs Batch

* **Batch**: A group of training examples processed together
* **Epoch**: One full pass over the entire training dataset

📌 Loss typically fluctuates per batch but should trend downward across epochs.

---

### 🔹 Train vs Validation

| Phase          | Purpose                    | Dropout/BNorm Active? |
| -------------- | -------------------------- | --------------------- |
| **Training**   | Learn via gradient descent | ✅ Yes                 |
| **Validation** | Monitor generalization     | ❌ No                  |

---

## PyTorch Full Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = MiniCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set to training mode
model.train()

for epoch in range(10):  # num_epochs
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}, Accuracy: {acc:.2f}%")
```

### 🔸 Add Validation

```python
model.eval()  # turn off Dropout & BatchNorm
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        val_loss = criterion(outputs, labels)
```

---

## TensorFlow Full Training Loop

```python
import tensorflow as tf

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

for epoch in range(10):
    print(f"\nEpoch {epoch + 1}")

    # TRAINING
    for images, labels in train_ds:
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc_metric.update_state(labels, predictions)

    train_acc = train_acc_metric.result()
    print(f"Training accuracy: {train_acc:.4f}")
    train_acc_metric.reset_state()

    # VALIDATION
    for val_images, val_labels in val_ds:
        val_preds = model(val_images, training=False)
        val_acc_metric.update_state(val_labels, val_preds)

    val_acc = val_acc_metric.result()
    print(f"Validation accuracy: {val_acc:.4f}")
    val_acc_metric.reset_state()
```

---

### 🔹 Saving Checkpoints

#### PyTorch

```python
torch.save(model.state_dict(), "checkpoint.pth")
model.load_state_dict(torch.load("checkpoint.pth"))
```

#### TensorFlow

```python
model.save_weights("checkpoint.h5")
model.load_weights("checkpoint.h5")
```

---

### 🔹 Early Stopping and Learning Rate Scheduling

#### PyTorch

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

# After each epoch:
scheduler.step(val_loss)
```

#### TensorFlow

```python
callback_list = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2)
]

model.fit(train_ds, validation_data=val_ds, callbacks=callback_list, epochs=10)
```

---

## Framework Comparison Table

| Feature          | PyTorch                             | TensorFlow                             |
| ---------------- | ----------------------------------- | -------------------------------------- |
| Forward pass     | `outputs = model(images)`           | `preds = model(images, training=True)` |
| Loss computation | `loss = criterion(outputs, labels)` | `loss_fn(labels, preds)`               |
| Backpropagation  | `loss.backward()`                   | `tape.gradient(...)`                   |
| Weight update    | `optimizer.step()`                  | `apply_gradients(...)`                 |
| Epoch logging    | Manual                              | Metrics + custom logging               |
| Early stopping   | Manual or `torch_lr_finder`         | Built-in `callbacks`                   |

---

## Mini-Exercise

Create a complete training loop for a 3-class classification task:

1. Use PyTorch or TensorFlow
2. Track:

   * Training loss
   * Training accuracy
   * Validation accuracy
3. Add:

   * Early stopping
   * Reduce LR on plateau
   * Model checkpoint saving

**Bonus**: Plot training/validation accuracy per epoch using `matplotlib`.

---