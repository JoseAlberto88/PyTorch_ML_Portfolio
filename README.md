# 🧠 PyTorch ML Portfolio  

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![GitHub](https://img.shields.io/badge/Status-Active-brightgreen)

*A collection of clean, modular, and well‑structured PyTorch projects demonstrating core machine learning skills.*

This repository showcases my progression through essential PyTorch concepts, including tensors, the ML workflow, image classification, and custom datasets.  
It includes both **notebooks for learning** and **modular Python scripts** for real‑world ML engineering.

---

## 🚀 Skills Demonstrated

- PyTorch fundamentals (tensors, autograd, modules)
- Building neural networks from scratch
- Training loops, evaluation, and metrics
- Image classification with CNNs
- Custom datasets and dataloaders
- Modular ML code structure (`src/models`, `src/utils`)
- Saving/loading models
- Clean GitHub project organization

---

## 📁 Project Structure

```
PyTorch_ML_Portfolio/
│
├── notebooks/              # Learning + experimentation
│   ├── 00_pytorch_fundamentals.ipynb
│   ├── 01_pytorch_workflow.ipynb
│   ├── 02_pytorch_classification.ipynb
│   └── 04_pytorch_custom_datasets.ipynb
│
├── src/
│   ├── models/
│   │   └── tinyvgg.py      # CNN model
│   ├── utils/
│   │   └── helpers.py      # Accuracy, saving, timing
│   └── train.py            # Training script
│
├── .gitignore
└── README.md
```

---

## 🏋️ Training Script

The `train.py` script trains a TinyVGG model on FashionMNIST:

```bash
python src/train.py
```

It handles:

- dataset loading  
- transforms  
- dataloaders  
- model creation  
- loss + optimizer  
- training loop  
- evaluation  
- model saving  

---

## 🧩 Model Architecture (TinyVGG)

The model is defined in:

```
src/models/tinyvgg.py
```

It includes:

- two convolutional blocks  
- ReLU activations  
- max pooling  
- a linear classifier  

---

## 📦 Utilities

Located in:

```
src/utils/helpers.py
```

Includes:

- `save_model()`  
- `accuracy_fn()`  
- `print_train_time()`  

---

## 🎯 Purpose of This Repository

This portfolio demonstrates my ability to:

- learn and apply PyTorch fundamentals  
- structure ML projects professionally  
- write clean, modular, reusable code  
- train and evaluate deep learning models  
- organize a GitHub repository for recruiters and collaborators  

---

## 📬 Contact

**José Alberto Martinez Morales**  
Machine Learning & Data Analytics  
Niagara Falls, ON, Canada
