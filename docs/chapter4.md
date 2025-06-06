---
hide:
    - toc
---

# **Chapter 4: Standard Image Preprocessing**

> “*A well-preprocessed image is half the training. The other half is just optimization.*”

---

## **Why This Chapter Matters**

Imagine handing a blurry, off-centered photo to a human and asking, “What’s in this?” You’d expect a confused answer—and that’s exactly how neural networks feel when they receive poorly scaled, misaligned, or inconsistent input images.

Preprocessing is the discipline of preparing your data in a way the model can truly understand. It’s not just resizing or flipping—it’s about standardizing, cleaning, and augmenting images so your CNN can extract meaningful patterns.

Done right, preprocessing:

  - Boosts model accuracy

  - Reduces overfitting

  - Makes inference stable

  - Speeds up convergence

Done wrong? You’ll spend weeks tuning your architecture for nothing.

---

## **Conceptual Breakdown**

**🔹 What Is Image Preprocessing?**

Image preprocessing refers to the transformations applied to raw image data before it’s fed into the model. It’s the very first thing your pipeline does, and its job is to:

  - Resize or crop the image to fit the model’s input size

  - Convert the pixel values to float and normalize them

  - Augment images with randomness during training to improve generalization

---

**🔹 Common Preprocessing Operations**

|Step	        |Purpose                                                        |
|---------------|---------------------------------------------------------------|
|Resize	        |Match input shape expected by CNN (e.g., 224×224)              |
|Crop	        |Focus on central content or apply randomness                   |
|Normalize	    |Scale values for model stability and consistency               |
|Augment	    |Random changes (flip, rotate, jitter) to generalize better     |

---

**🔹 Mean-Std Normalization vs 0–1 Scaling**

You’ll often see two types of normalization:

1. **0–1 scaling**: divide pixel values by 255

    - Simpler, used for custom models

2. **Mean-std normalization**: subtract dataset mean, divide by std

    - Used with pretrained models (e.g., ResNet, MobileNet)

|Format	        |Example                      |
|---------------|-----------------------------|
|0–1 Scaling	|`img = img / 255.0`          |
|Mean-std Norm	|`img = (img - mean) / std`   |
|ImageNet Mean	|`[0.485, 0.456, 0.406]`      |
|ImageNet Std	|`[0.229, 0.224, 0.225]`      |

---

**🔹 Effects of Preprocessing on Training**

|Preprocessing Problem	|Symptoms                           |
|-----------------------|-----------------------------------|
|No normalization	    |Model fails to converge            |
|Wrong mean/std	        |Bad predictions, poor transfer     |
|Mismatched resize	    |Shape errors at input layers       |
|Augmenting test data	|Erratic evaluation accuracy        |

Note: Always apply augmentation to training data only, never to validation/test.

---

**🔹 PIL vs OpenCV vs tf.image**

|Library	    |Style	        |Format Used            |
|---------------|---------------|-----------------------|
|PIL	        |Pythonic	    |RGB (default)          |
|OpenCV	        |Fast, C++	    |BGR (must convert!)    |
|tf.image	    |Tensor-native	|TensorFlow Tensors     |

If you’re mixing OpenCV with TensorFlow or PyTorch, be careful—color channels will be flipped unless you convert BGR → RGB.

---

**🔹 Preprocessing Matching: Train vs Inference**
Your model learns with a certain set of expectations (shape, scale, mean/std). If your inference pipeline doesn’t match your training pipeline, your model will:

Output low-confidence predictions

Misclassify even familiar data

> *📌 Golden Rule: Always reuse your training preprocessing (minus augmentation) for inference.*

---

## PyTorch Implementation

**🔸 Training Preprocessing**
```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # scales to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**🔸 Validation / Inference Preprocessing**
```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

## TensorFlow Implementation

**🔸 Training Preprocessing**
```python
import tensorflow as tf

def preprocess_train(img):
    img = tf.image.resize(img, [256, 256])
    img = tf.image.random_crop(img, [224, 224, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.cast(img, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img
```

---

**🔸 Inference Preprocessing**
```python
def preprocess_eval(img):
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img
```

---

## **Framework Comparison Table**

|Task	                    |PyTorch	                        |TensorFlow                             |
|---------------------------|-----------------------------------|---------------------------------------|
|Resize	                    |transforms.Resize()	            |tf.image.resize()                      |
|Crop	                    |RandomCrop, CenterCrop	            |tf.image.random_crop()                 |
|Flip	                    |RandomHorizontalFlip()	            |tf.image.random_flip_left_right()      |
|Normalize	                |transforms.Normalize(mean, std)	|Manual: (img - mean) / std             |
|Convert to tensor	        |transforms.ToTensor()	            |tf.cast(img, tf.float32) / 255.0       |
|Augment only in training	|Manual via Compose	                |Wrap in @tf.function or dataset map    |

---

## M**ini-Exercise**

Goal: Create a preprocessing function for training and one for inference.

1. Pick an image of any size (e.g., 500×300).

2. Apply:

    - Resize to 256×256

    - RandomCrop to 224×224 (training only)

    - Horizontal flip (training only)

    - Normalize using ImageNet mean/std

3. Compare preprocessed training and inference outputs.

4. Visualize the results using matplotlib.

Bonus: Try using OpenCV to load the image and manually convert from BGR to RGB.

---
