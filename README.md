# Machine Learning Lab 4 – K-Nearest Neighbors Classification

## Overview

This project demonstrates the implementation of the **K-Nearest Neighbors (KNN) Classification** algorithm using **Python** and **Scikit-learn**.

The model is trained and evaluated using the **Iris dataset**, a well-known dataset used for demonstrating classification algorithms. KNN classifies an unseen data point based on the classes of its nearest neighboring data points.

The project covers data loading, exploratory visualization, feature scaling, model training, prediction, and evaluation.

---

## Objectives

* Understand the working principle of the **K-Nearest Neighbors algorithm**.
* Load and explore the Iris dataset.
* Visualize relationships between different features.
* Separate input features and target labels.
* Split the dataset into training and testing sets.
* Standardize the input features.
* Train a KNN classification model.
* Predict the classes of unseen test data.
* Evaluate model performance using classification accuracy.

---

## Dataset

The project uses the **Iris dataset**, which contains measurements of iris flowers belonging to three different species.

### Features

| Feature        | Description         |
| -------------- | ------------------- |
| `Sepal Length` | Length of the sepal |
| `Sepal Width`  | Width of the sepal  |
| `Petal Length` | Length of the petal |
| `Petal Width`  | Width of the petal  |

### Target Classes

The model classifies flowers into three species:

* **Setosa**
* **Versicolor**
* **Virginica**

The Iris dataset contains **150 samples** with **4 numerical features**.

---

## Technologies Used

* **Python 3.x**
* **Jupyter Notebook / Google Colab**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

---

## Project Structure

```text
Machine-Learning_Lab4/
│
├── B_B1_14_Shreya_Sharma_ml_lab4.ipynb
├── iris.csv
└── README.md
```

---

## Workflow

The project follows these steps:

### 1. Import Required Libraries

The necessary Python libraries are imported for data manipulation, visualization, and machine learning.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Scikit-learn is used for dataset preparation, scaling, model training, and evaluation.

---

### 2. Load the Dataset

The Iris dataset is loaded and inspected to understand its structure.

The dataset contains flower measurements along with their corresponding species.

---

### 3. Explore the Dataset

The dataset is explored using functions such as:

```python
df.head()
df.info()
df.describe()
```

This helps understand:

* Number of observations
* Feature types
* Statistical properties
* Target classes

---

### 4. Data Visualization

Different plots are used to visualize relationships between the Iris features.

Examples include:

* Scatter plots
* Pair plots
* Feature distributions
* Class-wise visualizations

Visualization helps identify patterns and understand how the three flower species differ from one another.

---

### 5. Separate Features and Target

The dataset is divided into:

* **X:** Input features
* **y:** Target class

The four flower measurements are used as input features, while the species is used as the target.

---

### 6. Train-Test Split

The dataset is divided into training and testing data.

The training set is used to build the KNN model, while the testing set is used to evaluate how well the model performs on unseen data.

---

### 7. Feature Scaling

KNN is a distance-based algorithm. Therefore, feature scaling is important because features with larger numerical ranges can otherwise have a greater influence on distance calculations.

`StandardScaler` from Scikit-learn is used to standardize the features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

# K-Nearest Neighbors Algorithm

## What is KNN?

**K-Nearest Neighbors (KNN)** is a supervised machine learning algorithm used for classification and regression.

For classification, KNN determines the class of a new data point by looking at its **K nearest training examples** and assigning the class that occurs most frequently among those neighbors.

### Basic Process

1. Choose the value of **K**.
2. Calculate the distance between the new data point and training points.
3. Identify the K nearest neighbors.
4. Examine their class labels.
5. Assign the majority class to the new data point.

---

## Model Training

The KNN classifier is implemented using Scikit-learn:

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)
```

The model uses **5 nearest neighbors** for classification.

---

## Prediction

After training, predictions are made on the test dataset:

```python
y_pred = knn.predict(X_test)
```

The predicted classes are compared with the actual classes to evaluate the model.

---

## Model Evaluation

The model performance is evaluated using **classification accuracy**.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

Accuracy represents the proportion of correctly classified test samples.

---

## Key Concepts Demonstrated

### Distance-Based Classification

KNN uses the distance between data points to determine their similarity.

### Value of K

The value of `K` determines how many neighboring points are considered during classification.

* Small K → More sensitive to individual points
* Large K → Smoother decision boundaries

### Feature Scaling

Scaling ensures that all features contribute fairly to distance calculations.

### Supervised Learning

KNN learns from labeled training examples and uses those examples to classify new observations.

---

## Advantages of KNN

* Simple and easy to understand.
* Easy to implement.
* No complex training process.
* Can work well on small datasets.
* Useful for classification problems.

---

## Limitations of KNN

* Prediction can be computationally expensive for large datasets.
* Sensitive to the choice of K.
* Sensitive to feature scaling.
* Performance can decrease when there are many irrelevant features.
* Requires storing the training data.

---

## Applications

KNN can be used for:

* Pattern recognition
* Image classification
* Recommendation systems
* Customer classification
* Medical diagnosis
* Handwriting recognition
* Similarity-based search

---

## Output

The project produces:

* Iris dataset preview
* Dataset information and statistics
* Feature visualizations
* Training and testing datasets
* Scaled feature values
* Trained KNN classification model
* Predicted class labels
* Model accuracy

---

## Libraries Required

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Shreyaa-spax/Machine-Learning_Lab4.git
```

### 2. Open the project directory

```bash
cd Machine-Learning_Lab4
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Open the notebook

```bash
jupyter notebook
```

Open the Lab 4 notebook and run the cells sequentially.

The notebook can also be uploaded and executed using **Google Colab**.

---

## Conclusion

This project demonstrates the implementation of the **K-Nearest Neighbors classification algorithm** using the Iris dataset.

The project covers the complete basic machine learning workflow, including dataset exploration, visualization, feature preparation, scaling, model training, prediction, and evaluation.

KNN provides an intuitive example of how distance and similarity between data points can be used to perform classification.

---

## Author

**Shreya Sharma**

B.Tech Computer Science Engineering (AI & ML)

Ramdeobaba University, Nagpur

---

## License

This project is created for **educational and academic purposes** as part of a Machine Learning Laboratory assignment.
