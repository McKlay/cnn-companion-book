---
hide:
    - toc
---

# **Appendices**

---

## **Appendix A. PyTorch vs TensorFlow Cheatsheet**

This quick reference is designed to help you **switch between frameworks** with confidence. It highlights the **syntax, structure, and behavioral differences** that you need to remember during development.

| Task                  | PyTorch                              | TensorFlow                                  |
| --------------------- | ------------------------------------ | ------------------------------------------- |
| **Import**            | `import torch`                       | `import tensorflow as tf`                   |
| **Tensor Creation**   | `torch.tensor([[1., 2.], [3., 4.]])` | `tf.constant([[1., 2.], [3., 4.]])`         |
| **Shape Format**      | `[C, H, W]`                          | `[H, W, C]`                                 |
| **Reshape**           | `tensor.view(-1, 3, 32, 32)`         | `tf.reshape(tensor, [-1, 32, 32, 3])`       |
| **Permute/Transpose** | `x.permute(0, 2, 3, 1)`              | `tf.transpose(x, [0, 2, 3, 1])`             |
| **Device Handling**   | `x.to("cuda")`                       | Automatically uses GPU if available         |
| **Model Definition**  | Subclass `nn.Module`                 | Subclass `tf.keras.Model`                   |
| **Forward Pass**      | `def forward(self, x):`              | `def call(self, inputs):`                   |
| **Loss Functions**    | `nn.CrossEntropyLoss()`              | `tf.keras.losses.CategoricalCrossentropy()` |
| **Optimizers**        | `torch.optim.Adam()`                 | `tf.keras.optimizers.Adam()`                |
| **Training Mode**     | `model.train()` / `model.eval()`     | `training=True` / `False`                   |
| **Disable Gradients** | `with torch.no_grad():`              | `@tf.function` or `with tf.GradientTape():` |
| **Checkpointing**     | `torch.save(model.state_dict())`     | `model.save_weights()`                      |

*Use this table anytime you're translating code or debugging between frameworks.*

---

## **Appendix B. Troubleshooting Image Model Failures**

Training an image classifier but it keeps failing? Here’s a **diagnostic checklist**.

### ⚠️ Symptom: Model Predicts Only One Class

* Check normalization: are pixel values normalized properly (0–1 or mean-std)?
* Inspect label imbalance—does one class dominate the training data?
* Try a confusion matrix to confirm.

### ⚠️ Symptom: Shape Mismatch Errors

* Inspect input shape: is `[C, H, W]` (PyTorch) or `[H, W, C]` (TensorFlow)?
* Add `.unsqueeze(0)` or `expand_dims()` if batch dim is missing.
* Is the model expecting 3 channels? Grayscale images may only have 1.

### ⚠️ Symptom: Loss Not Decreasing

* Use `model.train()` (PyTorch) or `training=True` (TensorFlow).
* Check learning rate: too high causes instability, too low slows learning.
* Are labels correctly encoded (integers for CrossEntropy, one-hot for categorical)?

### ⚠️ Symptom: Good Train Accuracy, Poor Val Accuracy

* Overfitting? Add dropout or L2 regularization.
* Add more data or augmentations.
* Match preprocessing between training and inference.

### Tip: Always Test with a Single Image First

```python
# PyTorch single image test
model.eval()
with torch.no_grad():
    out = model(image.unsqueeze(0).to(device))
```

---

## **Appendix C. Glossary of Key Terms**

A quick dictionary of the most important terms in CNN-based vision.

| Term                  | Definition                                                     |
| --------------------- | -------------------------------------------------------------- |
| **Tensor**            | A multi-dimensional array (e.g., 4D tensor for image batches). |
| **Convolution**       | A sliding window operation to extract spatial patterns.        |
| **Kernel / Filter**   | The small matrix (e.g., 3×3) used in convolutions.             |
| **Stride**            | Step size the filter moves each time.                          |
| **Padding**           | Adding borders to preserve spatial size after convolution.     |
| **Channel**           | Color or feature dimension (e.g., RGB = 3 channels).           |
| **Batch Size**        | Number of images processed in one forward/backward pass.       |
| **Pooling**           | Downsampling layer (MaxPool, AvgPool) to reduce dimensions.    |
| **Activation**        | Non-linearity applied after each layer (ReLU, Sigmoid).        |
| **Feature Map**       | Output of a convolutional layer representing features.         |
| **Fine-Tuning**       | Adapting a pretrained model on a new dataset.                  |
| **Freeze Layers**     | Prevent layers from updating during training.                  |
| **Transfer Learning** | Reusing a trained model’s knowledge in a new task.             |

---

## **Appendix D. Pretrained Model Reference Table**

Use this table to find the right pretrained model for your project and the correct preprocessing function.

| Model              | Framework  | Import                                 | Preprocess Function               | Notes               |
| ------------------ | ---------- | -------------------------------------- | --------------------------------- | ------------------- |
| **ResNet50**       | PyTorch    | `torchvision.models.resnet50()`        | `transforms.Normalize(mean, std)` | Default 224×224     |
|                    | TensorFlow | `tf.keras.applications.ResNet50`       | `preprocess_input`                |                     |
| **MobileNetV2**    | PyTorch    | `torchvision.models.mobilenet_v2()`    | `Normalize(mean, std)`            | Lightweight         |
|                    | TensorFlow | `tf.keras.applications.MobileNetV2`    | `preprocess_input`                |                     |
| **EfficientNetB0** | PyTorch    | `torchvision.models.efficientnet_b0()` | `Normalize(mean, std)`            | Good accuracy/speed |
|                    | TensorFlow | `tf.keras.applications.EfficientNetB0` | `preprocess_input`                |                     |
| **VGG16**          | PyTorch    | `torchvision.models.vgg16()`           | `Normalize(mean, std)`            | Large and old       |
|                    | TensorFlow | `tf.keras.applications.VGG16`          | `preprocess_input`                |                     |

🔗 *All models available at:*

* [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
* [TensorFlow Keras Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)

---

## **Appendix E. Sample Projects and Mini-Exercises per Chapter**

Each chapter includes a **practical checkpoint or challenge**. Here’s a recap to revisit later:

| Chapter                    | Exercise                                                                       |
| -------------------------- | ------------------------------------------------------------------------------ |
| **1**. Image Input         | Convert an image (JPG) to a 3D tensor and visualize its shape.                 |
| **2**. Tensors             | Practice reshaping, permuting, and broadcasting tensors.                       |
| **3**. Input Pipeline      | Write a full image → tensor → model input pipeline.                            |
| **4**. Preprocessing       | Visualize how normalization and resize affect your image.                      |
| **5**. Pretrained Models   | Try ResNet50 on a custom image and print predictions.                          |
| **6**. Datasets            | Load a folder of images using `ImageFolder` or `image_dataset_from_directory`. |
| **7**. Augmentation        | Apply at least 3 different augmentations and view the effects.                 |
| **8**. CNN Layers          | Build a custom `Conv2d → ReLU → MaxPool` pipeline manually.                    |
| **9**. CNN Terms           | Draw out how kernel, stride, and padding affect feature map size.              |
| **10**. Forward/Call       | Implement `forward()` in PyTorch or `call()` in TensorFlow.                    |
| **11**. Summary            | Inspect model parameters and freeze layers selectively.                        |
| **12**. Build CNN          | Build and train a LeNet-style CNN from scratch.                                |
| **13**. Training           | Write a full training loop or use `model.fit()`.                               |
| **14**. Fine-Tuning        | Load MobileNetV2, freeze base, and train a new classifier head.                |
| **15**. Generalization     | Implement early stopping and visualize overfitting.                            |
| **16**. Eval Mode          | Compare predictions in `train()` vs `eval()` mode.                             |
| **17**. Feature Maps       | Visualize intermediate layer outputs with hooks or submodels.                  |
| **18**. Inference Pipeline | Build a reusable inference pipeline that handles image → model prediction.     |
| **19**. Debug Checklist    | Pick a broken model and fix it using the debugging checklist.                  |

---

# Final Words

These appendices complete your toolkit. Whether you're writing training code from scratch, deploying models, or debugging strange edge cases, **this section will bring you back on track**.

**Next Steps?**

* Bookmark this cheatsheet.
* Try the mini-exercises if you skipped any.
* Build and deploy a real project using one of the pretrained models.

If you’ve reached this far—you’re not just reading a book. You’ve **trained your own mind like a neural net**: layer by layer, with careful activation.

Let’s keep building.

---
