# 👁️ Eye Disease Classifier

**RetinalNet** is a high-performance Deep Learning application designed to classify retinal fundus images into four clinical categories (**Cataract, Diabetic Retinopathy, Glaucoma, and Normal**). This project serves as a diagnostic aid to help prevent irreversible vision loss through early AI-driven screening.

The project features a comprehensive Jupyter Notebook for model development and a user-friendly **Gradio** web application for real-time diagnostic predictions.

---

## 🚀 Features

### 🧠 Advanced AI Architectures
* **Comparative Modeling:** Evaluation between a custom-built CNN and Transfer Learning architectures.
* **Custom CNN:** A 4-layer convolutional neural network designed to extract foundational ocular features.
* **EfficientNetB0:** Leverages state-of-the-art "Compound Scaling" and pre-trained ImageNet weights for maximum precision.
* **Intelligent Training:** Implements `ReduceLROnPlateau` and `EarlyStopping` to achieve optimal convergence and prevent overfitting.



### ✨ Modern Web Interface (Gradio)
* **Interactive UI:** Clean, responsive drag-and-drop interface for image uploads.
* **Instant Analysis:** Real-time categorization with confidence scores for each disease.
* **Probability Mapping:** Displays the model's certainty across all four categories for clinical transparency.

---

## 📂 Project Structure

```text
├── eye-disease-classification-custom-cnn-effnetb0.ipynb  # Data analysis, EDA, and Model Training
├── eye_disease_model2.h5                                
├── app.py                                              # Gradio Application Entry Point
├── requirements.txt                                    # Project dependencies
└── README.md                                           # Project Documentation

---


## 🛠️ Installation

### **Prerequisites**

* Python **3.8 or higher**
* **pip** package manager

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/ocularis-eye-classifier.git
cd ocularis-eye-classifier
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

---




## 🖥️ Usage

### **Running the Web App**

Launch the Gradio interface locally:

```bash
python app.py
```

The app will open automatically in your browser at:

```
http://127.0.0.1:7860
```

### **How to Use**

1. **Upload Image** – Drag and drop a retinal fundus image into the input area.
2. **Analyze** – The model automatically preprocesses the image (224×224) and performs inference.
3. **View Results** – See:

   * Predicted disease category
   * Confidence scores for all four eye conditions

---

## 📊 Model Performance

The models were trained using a high-quality retinal dataset. Below are the test results:

| Model Architecture | Test Accuracy | Test Loss  | Key Strength                               |
| ------------------ | ------------- | ---------- | ------------------------------------------ |
| **EfficientNetB0** | **89.10%**    | **0.2962** | Best for high-precision clinical diagnosis |
| **Custom CNN**     | **85.07%**    | **0.4210** | Lightweight and fast inference             |

More detailed evaluation and visual training history can be found in:
**`eye-disease-classification-custom-cnn-effnetb0.ipynb`**

---

## 🤝 Contributing

Contributions help make the open‑source community a powerful place to learn, inspire, and create. Any improvements you make to **Ocularis Eye Classifier** are greatly appreciated!

### **How to Contribute**

1. **Fork the Project**
2. **Create your Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit your Changes**

   ```bash
   git commit -m "Add some AmazingFeature"
   ```
4. **Push to the Branch**

   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request**

---

## 📝 License

This project is open-source and available under the **MIT License**.


