# Potato-Pepper
---

# 🌿 Deep Learning for Plant Disease Detection

> Motivation: *"4 years of mess food that contained potato mixed in every sabji sparked the idea for this project."*
> *This is a deep learning-based solution to a very real agricultural challenge: early detection of plant diseases using computer vision.*

---

## 🧠 Motivation

Crop diseases can cause huge losses, especially if detected late. Traditionally, farmers rely on experience and visual inspection, which is time-consuming and often inaccurate. With deep learning, we can automate the process and improve reliability.

In this project, I built two Convolutional Neural Network (CNN) models to detect:

* **Potato** diseases (Early and Late Blight)
* **Bell Pepper** disease (Bacterial Spot)

---

## 📂 Dataset

We used the **PlantVillage dataset** from Kaggle — a widely used, annotated dataset of plant leaves.

* **Potato**: 3 Classes (Early Blight, Late Blight, Healthy)
* **Bell Pepper**: 2 Classes (Bacterial Spot, Healthy)
* Over **2000 images**, augmented during preprocessing

---

## 🧱 Model Architecture

The models are built using **TensorFlow and Keras**. Here’s a breakdown:

1. **Preprocessing Layer**: Resize + Normalize
2. **Conv2D Layers**: Feature extraction (spots, edges)
3. **MaxPooling2D**: Downsampling to retain important features
4. **Flatten**: Convert to 1D vector
5. **Dense Layers**: Fully connected layer for learning
6. **Softmax Output**: Multi-class classification

![image](https://github.com/user-attachments/assets/58f066d6-cc61-4f27-a0c1-2e66bc792e78)


---

## ⚙️ Hyperparameters

* **Optimizer**: Adam
* **Learning Rate**: 0.001
* **Epochs**:

  * Bell Pepper: 15
  * Potato: 20

---

## 🧼 Data Handling

To speed up training and reduce memory load:

* Used **TensorFlow’s `cache()` and `prefetch()`** functions.
* Enabled **shuffling** to prevent overfitting.

[Watch this video to learn more](https://www.youtube.com/watch?v=MLEKEplgCas&t=5s)

---

## 🧪 Augmentation

We used Keras' built-in methods to apply:

* **Random Flipping**
* **Random Rotation**

This helped increase diversity in the dataset and reduce overfitting.

---

## 📈 Training & Results

* Dataset split: **80% Training / 10% Validation / 10% Testing**
* The models showed strong convergence with high accuracy and low validation loss.


---

## 🔍 Evaluation

Below are examples from the test set, with predicted classes and confidence scores.

✅ The model accurately classifies both healthy and diseased leaves for **potato** and **pepper** plants.


---

## 🧬 Multimodal Analysis (Research Insight)

I explored advanced research that uses **multimodal learning**, combining:

* **Image Data** (e.g., hyperspectral and thermal)
* **Text Data** (e.g., symptoms, field logs)

> 🔍 Researchers used **MSC-ResNet**, **MSC-TextCNN**, and **CT-CNN**, along with GPT-4 to develop a “Potato GPT” for intelligent disease diagnosis.

---

## 🚀 Deployment Direction

* On the left: UI for “Potato GPT”
* On the right: Confusion matrices of various models


📊 Multimodal models significantly outperform traditional CNNs — especially on **healthy crop identification**.

---

## ✅ Conclusion and Future Work

> Our CNN-based model provides fast, scalable, and accurate plant disease detection.

### 🔮 Future Goals:

* Add more plant types (e.g., Tomato – 10 class classification)
* Integrate multimodal learning for deeper insights
* Develop a complete **mobile/web app** for real-world deployment

---

## 📎 Resources

* [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
* [Netron Model Visualizer](https://netron.app)
* [TensorFlow Caching & Prefetching](https://www.youtube.com/watch?v=MLEKEplgCas&t=5s)

---

## 🙏 Thanks for Reading!
