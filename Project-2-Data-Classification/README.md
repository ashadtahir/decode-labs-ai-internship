# Project 2: Data Classification Using AI

**Decode Labs AI Internship**

## Objective

Build a basic classification model that predicts the species of an Iris flower based on four measurements. This project demonstrates a supervised learning classification pipeline using Python and scikit-learn.

## Dataset

* **Name:** Iris Dataset
* **Source:** Built into scikit-learn using `sklearn.datasets.load_iris`
* **Samples:** 150
* **Features:** 4

  * Sepal length
  * Sepal width
  * Petal length
  * Petal width
* **Classes:** 3

  * Setosa
  * Versicolor
  * Virginica
* **Class distribution:** 50 samples per class

## Technologies and Libraries

* Python 3
* scikit-learn

## Classification Pipeline

### 1. Load the Dataset

The Iris dataset is loaded using scikit-learn. It contains 150 samples, 4 features, and 3 flower classes.

### 2. Train/Test Split

The dataset is divided into:

* **80% training data:** 120 samples
* **20% testing data:** 30 samples

A `random_state` of 42 is used to make the split reproducible.

### 3. Feature Scaling

`StandardScaler` is fitted using the training data and then used to transform both the training and testing data.

This keeps the test data separate from the scaling process.

### 4. KNN Classification

A **K-Nearest Neighbors (KNN)** classifier is used with:

```text
K = 5
```

The model is trained using the scaled training data and then used to predict the classes of the test data.

### 5. Model Evaluation

The model is evaluated using:

* **Accuracy** — percentage of correct predictions
* **Confusion Matrix** — shows correct and incorrect predictions for each class
* **F1 Score** — combines precision and recall into a single score

## How to Install Dependencies

Make sure Python is installed, then run:

```bash
pip install scikit-learn
```

## How to Run

Navigate to the project folder:

```bash
cd Project-2-Data-Classification
```

Then run:

```bash
python classification.py
```

## Results

The final test run produced:

```text
Accuracy Score:  1.0000 (100.00%)
F1 Score:        1.0000

Confusion Matrix:
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

Total correct: 30/30
```

The model correctly classified all 30 test samples.

## Reflection

This project helped me understand the basic workflow of supervised machine learning.

Key things I learned:

* How to load and work with a dataset using scikit-learn
* Why training and testing data should be separated
* How feature scaling works with `StandardScaler`
* How KNN uses nearby data points to make predictions
* How to evaluate a classification model using accuracy, a confusion matrix, and F1 score

The Iris dataset produced a 100% accuracy and F1 score of 1.0000 in the final test run. This shows that the model classified the selected test samples correctly.

## Project Structure

```text
Project-2-Data-Classification/
├── classification.py
└── README.md
```
