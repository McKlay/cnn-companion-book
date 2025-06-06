---
hide:
  - toc
---

## **Vision in Code: Mastering Convolutional Neural Networks for Real-World Image Modeling**
### A Practical Guide to CNN Implementation with PyTorch and TensorFlow

---

### **Contents**

---

#### 📖 [Preface](Preface.md)

- [Why This Book Exists](Preface.md#why-this-book-exists)

- [Who Should Read This](Preface.md#who-should-read-this)

- [From Pixels to Convolutions: How This Book Was Born](Preface.md#from-pixels-to-convolutions-how-this-book-was-born)

- [What You’ll Learn (and What You Won’t)](Preface.md#what-youll-learn-and-what-you-wont)

- [How to Read This Book (Even if You’re Just Starting Out)](Preface.md#how-to-read-this-book-even-if-youre-just-starting-out)

---

#### Part I – [Foundations of Image Tensors and Preprocessing](PartI_overview.md)

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 1: [How a Neural Network Sees an Image](chapter1.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1 What is an image (JPEG, PNG, etc.) in memory?

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.2 From pixel data → NumPy array → Tensor

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.3 RGB channels, 8-bit scale, float conversion

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.4 [H, W, C] vs [C, H, W] — framework differences explained

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.5 Why model input shape matters

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.6 Visual walkthrough of image-to-input pipeline

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 2: [What is a Tensor (in Code and in Mind)?](chapter2.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1 Tensor shapes and memory layout

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2 Dimensionality intuition

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3 PyTorch: torch.tensor, .permute(), .view(), .reshape()

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.4 TensorFlow: tf.Tensor, .reshape(), .transpose(), broadcasting

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.5 Visual walkthroughs of shape manipulations

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 3: [From Pixels to Model Input](chapter3.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1 Full image input pipeline:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2 RGB loading → float32 conversion → normalization

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3 Resizing and reshaping to expected input size

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.4 Batch dimension handling: unsqueeze() vs expand_dims()

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.5 Feeding tensors into Dense or Conv2D layers

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.6 Debugging mismatched shapes

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.7 Framework comparison of entire image → tensor → model flow

---

#### Part II – [Preprocessing and Input Pipelines](PartII_overview.md)

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 4: [Standard Image Preprocessing](chapter4.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.1 Resize, Normalize, Augment

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.2 Mean-std normalization vs 0–1 scaling

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.3 Format mismatches and their impact on accuracy

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.4 PIL vs OpenCV vs tf.image

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.5 Visualizing preprocessing effects

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.6 Matching preprocessing between training and inference

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 5: [Preprocessing for Pretrained Models](chapter5.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.1 Matching pretrained model expectations: MobileNetV2, EfficientNet, ResNet, etc.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.2 transforms.Normalize vs tf.keras.applications.*.preprocess_input()

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.3 PyTorch: torchvision.models

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.4 TensorFlow: keras.applications

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.5 Inference vs training preprocessing pitfalls

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.6 Side-by-side code snippets for each model

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 6: [Image Datasets: Getting Data Into the Network](chapter6.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.1 Folder structure conventions

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.2 PyTorch: Dataset, DataLoader, ImageFolder, transforms

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.3 TensorFlow: tf.data.Dataset, image_dataset_from_directory()

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.4 Label mapping, batching, and shuffling

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.5 Visualizing batches from both frameworks

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 7: [Data Augmentation Techniques (Expanded)](chapter7.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.1 Common augmentations: RandomCrop, ColorJitter, Cutout

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.2 Advanced augmentations: Mixup, CutMix (optional)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.3 PyTorch: torchvision.transforms

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.4 TensorFlow: tf.image, Keras preprocessing layers

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.5 Before/after visualization of augmentation effects

---

#### Part III – [CNN Architectures and Concepts](PartIII_overview.md)

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 8: [Understanding CNN Layers](chapter8.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8.1 Kernels, filters, channels, strides, padding

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8.2 Pooling (Max, Average), ReLU, BatchNorm

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8.3 PyTorch: nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d, etc.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8.4 TensorFlow: Conv2D, MaxPooling2D, BatchNormalization, etc.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8.5 Conceptual breakdown + syntax comparison

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 9: [The CNN Vocabulary (Terms Demystified)](chapter9.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9.1 Key terms: kernel, convolution, stride, padding

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9.2 Input/output channels, feature maps

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9.3 Convolutional layer vs residual block

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9.4 Layer variants: ReflectionPad2d, InstanceNorm2d, AdaptiveAvgPool2d

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9.5 Visual and code-based examples

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 10: [Writing the forward() / call() Function](chapter10.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10.1 PyTorch: forward(), self.features, self.classifier

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10.2 TensorFlow: call(), subclassing Model

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10.3 Layer-by-layer flow visualized

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10.4 Common mistakes in model building

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 11: [Model Summary and Parameter Inspection](chapter11.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;11.1 PyTorch: model.parameters(), summary(), state_dict()

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;11.2 TensorFlow: .summary(), get_weights(), trainable_variables

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;11.3 How to freeze/unfreeze layers for fine-tuning

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 12: [Building Your First CNN: Patterns and Pitfalls](chapter12.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;12.1 Simple architectures: LeNet-style, Mini-VGG

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;12.2 Choosing filter sizes, kernel shapes, stride

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;12.3 Stacking layers: when and why

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;12.4 Common design mistakes (too few filters, wrong input shape, etc.)

---

#### Part IV – [Training and Fine-Tuning](PartIV_overview.md)

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 13: [Loss Functions and Optimizers](chapter13.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;13.1 PyTorch: loss_fn(), .backward(), optimizer.step()

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;13.2 TensorFlow: GradientTape, optimizer.apply_gradients()

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;13.3 Common losses: CrossEntropy

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;13.4 Optimizers: SGD, Adam

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;13.5 Visualizing gradient flow

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 14: [Training Loop Mechanics](chapter14.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;14.1 PyTorch: full training loop with train_loader

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;14.2 TensorFlow: model.fit() vs custom training loop

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;14.3 Logging loss and metrics

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;14.4 Checkpoint saving, early stopping

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;14.5 Adding visuals for debugging and learning

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 15: [Training Strategies and Fine-Tuning Pretrained CNNs](chapter15.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;15.1 When to Fine-Tune vs Freeze

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;15.2 Adapting Pretrained Models

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;15.3 Regularization Techniques

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;15.4 Training Strategies for Generalization

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;15.5 Recognizing Overfitting and Underfitting

---

#### Part V – [Inference, Evaluation, and Visual Debugging](PartV_overview.md)

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 16: [Train vs Eval Mode](chapter16.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;16.1 PyTorch: model.train(), model.eval(), no_grad()

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;16.2 TensorFlow: training=True/False

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;16.3 Dropout and BatchNorm behavior

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;16.4 Impact of mode on inference

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 17: [Visualizing Feature Maps and Filters](chapter17.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;17.1 Getting intermediate layer outputs

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;17.2 PyTorch: forward hooks, manual slicing

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;17.3 ensorFlow: defining sub-models

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;17.4 Visualizing what the model is focusing on

---

#### Part VI – [Deployment-Ready Insights](PartVI_overview.md)

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 18: [Inference Pipeline Design](chapter18.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;18.1 Keeping preprocessing consistent (train vs inference)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;18.2 Reusable preprocess functions

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;18.3 Input validation, test-time augmentation

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 19: [Common Errors and How to Debug Them](chapter19.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;19.1 Model always predicts one class? Check normalization

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;19.2 Input shape mismatch? Check dataloader

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;19.3 Nothing’s working? Try a single image pipeline

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;19.4 Debugging checklist for CNN-based models

---

#### [Appendices](appendices.md)

A. PyTorch vs TensorFlow Cheatsheet

B. Troubleshooting Image Model Failures

C. Glossary of Key Terms

D. Pretrained Model Reference Table (with links)

E. Sample Projects and Mini-Exercises per Chapter

---

#### Chapter Format

Each chapter ends with:

- Conceptual Breakdown

- PyTorch Implementation

- TensorFlow Implementation

- Framework Comparison Table

- Use Case or Mini-Exercise

---