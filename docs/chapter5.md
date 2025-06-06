---
hide:
    - toc
---

# **Chapter 5: Preprocessing for Pretrained Models**

> “*You can’t transfer knowledge if the inputs don’t speak the same language. Preprocessing isn’t optional—it’s the protocol.*”

---

## **Why This Chapter Matters**

Pretrained CNNs—like those from torchvision.models or keras.applications—have been trained on massive datasets like ImageNet. But they were trained with specific input formats in mind:

  - A fixed input shape (often 224×224)

  - Normalized using specific mean and std

  - RGB ordering (not grayscale or BGR)

  - Scaled values in a specific range (e.g., [0, 1], [-1, 1])

If you deviate from this, even slightly, your fine-tuned model might:

  - Output garbage predictions

  - Fail to converge during training

  - Seem to overfit instantly

This chapter teaches you how to match preprocessing exactly to each pretrained model so you can extract their full power—safely.

---

## **Conceptual Breakdown**

**🔹 Pretrained Model Expectations**

|Model	        |Expected Input Shape	  |Pixel Range	    |Normalization                          |
|---------------|-----------------------|-----------------|---------------------------------------|
|ResNet, VGG	  |224×224×3	            |[0.0 – 1.0]	    |Mean: `[0.485, 0.456, 0.406]` <br> Std: `[0.229, 0.224, 0.225]`  |
|MobileNetV2	  |224×224×3	            |[-1.0 – 1.0]	    |`preprocess_input()` scales it         |
|EfficientNet	  |224×224×3 or 240×240	  |[0.0 – 255.0]	  |`preprocess_input()` handles it        |

> ***Rule of thumb: Always check the docs or source code of the model you’re using.***

---

**🔹 Why You Can’t Just Use .ToTensor() or /255.0**

Because pretrained models were trained on data that was already:

  - Normalized using dataset-wide statistics

  - Possibly scaled to [-1, 1] or whitened

  - Fed in a specific channel order

If you skip or mismatch the normalization, you're effectively corrupting the input distribution—and the model’s learned filters won’t match.

---

**🔹 Matching PyTorch Pretrained Models**

Most PyTorch models in `torchvision.models` use:

  - `[0, 1]` range from `ToTensor()`

  - Mean-std normalization with ImageNet stats

```python
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
```

> 📌 Always use this transform after `ToTensor()`.

---

**🔹 Matching TensorFlow/Keras Pretrained Models**

Each model has a helper function:

|Model Family	  |Preprocessing Function                                 |
|---------------|-------------------------------------------------------|
|ResNet, VGG	  |`keras.applications.resnet50.preprocess_input()`       |
|MobileNetV2	  |`keras.applications.mobilenet_v2.preprocess_input()`   |
|EfficientNetB0	|`keras.applications.efficientnet.preprocess_input()`   |

These handle:

  - Mean/std normalization

  - Scaling to [-1, 1] if needed

  - RGB channel order

> 📌 These functions expect raw pixel values (0–255 float), not normalized.

---

## **PyTorch Implementation**

```python
from torchvision import models, transforms
from PIL import Image

# Load pretrained ResNet
model = models.resnet50(pretrained=True)
model.eval()

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # scales to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load image and preprocess
img = Image.open("dog.jpg").convert("RGB")
tensor = preprocess(img).unsqueeze(0)  # shape: [1, 3, 224, 224]
```

---

## **TensorFlow Implementation**

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pretrained MobileNetV2
model = MobileNetV2(weights="imagenet")

# Load and preprocess
img = image.load_img("dog.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)  # shape: [224, 224, 3]
img_array = np.expand_dims(img_array, axis=0)  # [1, 224, 224, 3]
img_array = preprocess_input(img_array)  # scales to [-1, 1]
```

---

## **Framework Comparison Table**

|Framework	        |Step	                          |PyTorch	                  |TensorFlow                               |
|-------------------|-------------------------------|---------------------------|-----------------------------------------|
|Preprocessing	    |Built-in model normalization	  |transforms.Normalize()	    |keras.applications.*.preprocess_input()  |
|Channel Order	    |Input Format	                  |[C, H, W]	                |[H, W, C]                                |
|Expected Range	    |Pixel Values	                  |[0, 1] + mean/std	        |[0, 255] → auto-scaled internally        |
|Input Shape	      |Default for pretrained	        |[1, 3, 224, 224]	          |[1, 224, 224, 3]                         |

---

## **Mini-Exercise**

Try loading the same image in both PyTorch and TensorFlow. Use:

  - `resnet50` in PyTorch

  - `ResNet50` in Keras

1. Apply the correct preprocessing for each framework.

2. Feed the tensor into the model and extract the top-1 class prediction.

3. Confirm both models give similar results.

Bonus: Print the difference between preprocessed tensors in PyTorch vs TensorFlow (after matching shape order).

---
