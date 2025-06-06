---
hide:
    - toc
---

# **Chapter 6: Image Datasets – Getting Data Into the Network**

> “*It’s not just about images—it’s about structure, batching, and flow. A CNN needs data, not chaos.*”

---

## **Why This Chapter Matters**

You’ve learned how to preprocess a single image, but deep learning thrives on batches of thousands of images—efficiently streamed, augmented, shuffled, and labeled.

If your dataset isn’t:

  - Properly structured

  - Mapped to class labels

  - Efficiently loaded and shuffled

… then even the best model won't help you. You’ll face:

  - GPU idle time

  - Overfitting from data leaks

  - Mislabeling issues

This chapter shows how to transform folders of images into usable model inputs, with complete control over batching, shuffling, label handling, and augmentation.

---

## **Conceptual Breakdown**

**🔹 The Canonical Folder Structure**

Most frameworks expect images arranged like this:
```kotlin
dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   ├── class2/
│   │   ├── img3.jpg
│   │   ├── img4.jpg
├── val/
│   ├── class1/
│   ├── class2/
```
Each folder name becomes the class label, and all images inside are automatically mapped.

This structure is:

  - Easy to understand

  - Supported natively in both PyTorch and TensorFlow

  - Flexible for augmentation and splitting

---

**🔹 Dataset Terms You Must Know**

|Term	        |Definition                                             |
|---------------|-------------------------------------------------------|
|Dataset	    |Collection of images and labels (can be lazy-loaded)   |
|Dataloader	    |Handles batching, shuffling, and parallel loading      |
|Transform	    |Applied per sample to augment/normalize                |
|Batch	        |A group of N samples fed to the model at once          |
|Epoch	        |One full pass through the dataset                      |

---

**🔹 Why Shuffling Matters**

Shuffling prevents:

 - Learning label order bias (e.g., all cats, then all dogs)

 - Memorizing batch patterns

 - Overfitting on dataset structure

> 📌 Always shuffle during training, not validation.

---

**🔹 Visualizing a Batch**

Understanding shapes is critical.

|Description	        |PyTorch Shape	    |TensorFlow Shape   |
|-----------------------|-------------------|-------------------|
|One RGB image	        |[3, H, W]	        |[H, W, 3]          |
|One batch (N images)	|[N, 3, H, W]	    |[N, H, W, 3]       |
|One batch + label	    |(images, labels)	|(images, labels)   |

## **PyTorch Implementation**

**🔸 Folder-based Dataset Loading**

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder("data/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Check class mapping
print(train_dataset.class_to_idx)  # e.g., {'cat': 0, 'dog': 1}
```

**🔸 Iterating Through a Batch**

```python
for images, labels in train_loader:
    print(images.shape)  # [32, 3, 224, 224]
    print(labels)        # Tensor of shape [32]
    break
```

---

## **TensorFlow Implementation**

**🔸 Load from Folder with `image_dataset_from_directory`**

```python
import tensorflow as tf

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),
    batch_size=32,
    label_mode="int",  # or 'categorical' for one-hot
    shuffle=True
)

# Check output
for images, labels in train_ds.take(1):
    print(images.shape)  # (32, 224, 224, 3)
    print(labels.shape)  # (32,)
```

**🔸 Apply Preprocessing via `.map()`**

```python
def preprocess(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img, label

train_ds = train_ds.map(preprocess)
```

---

## **Framework Comparison Table**

|Feature	                |PyTorch	                        |TensorFlow                                                 |
|---------------------------|-----------------------------------|-----------------------------------------------------------|
|Dataset Loader	            |`datasets.ImageFolder`	            |`tf.keras.preprocessing.image_dataset_from_directory`      |
|Custom Transforms	        |`transforms.Compose([...])`	    |`.map(preprocess_fn)`                                      |
|Shuffling	                |`shuffle=True` in DataLoader	    |`shuffle=True` in loader or `.shuffle(buffer)`             |
|Label format	            |`class_to_idx` dictionary	        |Auto-mapped, label mode configurable                       |
|Image shape in batch	    |`[B, 3, H, W]`	                    |`[B, H, W, 3]`                                             |

---

## **Mini-Exercise**

Prepare a small dataset with 2–3 image classes (e.g., cat, dog, bird). Then:

1. Use both PyTorch and TensorFlow to:

    - Load dataset from folder

    - Apply resizing and normalization

    - Create dataloaders/batched datasets

2. Iterate through 1 batch and print:

    - Image tensor shape

    - Label mapping (class-to-index)

3. Try visualizing one batch using matplotlib.

    - Bonus: Add a simple augmentation (flip or crop) and re-inspect the batch output.

---

