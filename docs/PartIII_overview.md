---
hide:
    - toc
---

# Part III – CNN Architectures and Concepts

> *“The data flows in—but what happens next? This is where filters activate, features emerge, and vision becomes understanding.”*

---

## Why This Part Matters

You’ve resized, normalized, and batched your image data.

But how does a neural network actually **see a dog’s ear**, **detect a road sign**, or **recognize a handwritten digit**?

It happens through the architecture: the convolutional layers, activations, pooling operations, and more that together form a **trainable perception system**. This part walks you through that internal machinery.

By the end of Part III, you’ll be able to:

* **Build CNN architectures from scratch**
* **Understand how each layer transforms the input**
* **Spot design patterns used in popular models**
* **Interpret what each layer contributes to learning**

---

## What You’ll Master in This Part

* The anatomy of CNN layers—what they do and how to use them
* Common terms like kernel, stride, padding, feature map, and filter
* The logic behind forward passes and `forward()`/`call()` methods
* How to inspect a model’s structure and parameter count
* CNN design patterns used in real-world architectures like LeNet and Mini-VGG

Each chapter dives deeper into the building blocks, visualizes internal flows, and compares implementation across **PyTorch** and **TensorFlow**—so you gain full control over how your models are structured and trained.

---

## Chapter Breakdown

| Chapter | Title                                            | What You’ll Learn                                                               |
| ------- | ------------------------------------------------ | ------------------------------------------------------------------------------- |
| **8**   | *Understanding CNN Layers*                       | Kernels, strides, padding, pooling, activations, and normalization layers       |
| **9**   | *The CNN Vocabulary (Terms Demystified)*         | What filters, channels, and feature maps really mean—visually and in code       |
| **10**  | *Writing the forward() / call() Function*        | Model subclassing, layer flow, common mistakes, debug-friendly architectures    |
| **11**  | *Model Summary and Parameter Inspection*         | Counting parameters, freezing/unfreezing layers, weight inspection              |
| **12**  | *Building Your First CNN: Patterns and Pitfalls* | Hands-on with LeNet, Mini-VGG, filter size strategies, design mistakes to avoid |

---

## Key Ideas That Tie This Part Together

1. **Layers aren’t black boxes—they’re geometric transformers.**

   * Each convolution shrinks or expands your input in predictable ways. Knowing how lets you *design with intention*.

2. **Your model is a blueprint.**

   * Whether you’re subclassing in PyTorch or TensorFlow, you need to know how tensors flow from layer to layer—and why.

3. **Architecture ≠ Accuracy—but it enables it.**

   * Fancy models don’t guarantee results, but *clear, modular, well-planned architectures* make success reproducible.

---

## What Makes This Part Unique

Most books or tutorials show a pre-written CNN, but they rarely:

* **Walk you through layer-by-layer decisions**
* Explain what happens **visually** inside each operation
* Compare `Conv2d` vs `Conv2D` with actual shape flows
* Show what happens when you pass an image into your `forward()` method

This part does.

---

## Learning Goal for This Part

By the end of Part III, you should be able to:

* Construct custom CNNs for new tasks
* Modify and fine-tune pretrained architectures
* Understand internal layer behavior and diagnose structural bugs
* Experiment confidently with kernel sizes, channel depths, and activation strategies

---
