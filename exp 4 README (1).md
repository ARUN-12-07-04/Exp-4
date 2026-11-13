# Experiment 5: Support Vector Machine (SVM)

## Aim

To implement and analyze the performance of a Support Vector Machine
(SVM) model for classification using Python.

## Overview

A Support Vector Machine is a supervised machine-learning algorithm used
for classification. It finds a decision boundary (hyperplane) that
separates classes while maximizing the margin between the closest data
points, called **support vectors**.

This experiment uses the **Iris dataset** and compares two SVM kernels:

-   **Linear kernel** -- suitable when classes can be separated
    approximately by a straight decision boundary.
-   **RBF (Radial Basis Function) kernel** -- can model non-linear
    decision boundaries.

For visualization, the experiment uses the first two features of the
Iris dataset.

## Algorithm

1.  **Start.**
2.  Load the Iris dataset and select the first two features for
    visualization.
3.  Split the data into training and testing sets.
4.  Standardize the features using `StandardScaler`.
5.  Train an SVM classifier with a linear kernel.
6.  Predict the test labels and calculate the classification accuracy.
7.  Train another SVM classifier with the RBF kernel.
8.  Predict the test labels and calculate the classification accuracy.
9.  Plot the decision boundaries for both models.
10. **Stop.**

## Mathematical Background

### Maximum-margin hyperplane

The SVM aims to maximize the margin between the separating hyperplane
and the nearest training samples. For a linearly separable dataset, the
optimization objective can be written as:

\[ `\max`{=tex}\_{w,b} `\frac{2}{\|w\|}`{=tex} \]

subject to the classification constraints:

\[ y_i(w\^T x_i+b)`\geq 1`{=tex} \]

### RBF kernel

The Radial Basis Function kernel is defined as:

\[
K(x_i,x_j)=`\exp`{=tex}`\left`{=tex}(-`\gamma`{=tex}\|x_i-x_j\|\^2`\right`{=tex})
\]

Here, `gamma` controls how far the influence of an individual training
example reaches.

## Requirements

-   Python 3
-   NumPy
-   Matplotlib
-   scikit-learn

Install the required packages:

``` bash
pip install numpy matplotlib scikit-learn
```

## Python Implementation

Save the following program as `svm_iris.py`.

``` python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Step 1: Load the Iris dataset and select the first two features
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Function to plot the decision boundary
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, predictions, alpha=0.25, cmap="viridis")
    scatter = plt.scatter(
        X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="viridis"
    )
    plt.xlabel("Standardized sepal length")
    plt.ylabel("Standardized sepal width")
    plt.title(title)
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=list(iris.target_names),
        title="Iris class"
    )
    plt.tight_layout()
    plt.show()


# Step 4: Train and evaluate the SVM with a linear kernel
linear_svm = SVC(kernel="linear")
linear_svm.fit(X_train_scaled, y_train)

linear_predictions = linear_svm.predict(X_test_scaled)
linear_accuracy = accuracy_score(y_test, linear_predictions)

print("Linear Kernel Accuracy:", round(linear_accuracy * 100, 2), "%")


# Step 5: Train and evaluate the SVM with an RBF kernel
rbf_svm = SVC(kernel="rbf", gamma="scale")
rbf_svm.fit(X_train_scaled, y_train)

rbf_predictions = rbf_svm.predict(X_test_scaled)
rbf_accuracy = accuracy_score(y_test, rbf_predictions)

print("RBF Kernel Accuracy:", round(rbf_accuracy * 100, 2), "%")


# Step 6: Plot decision boundaries using the training data
plot_decision_boundary(
    linear_svm, X_train_scaled, y_train,
    "SVM Decision Boundary - Linear Kernel"
)

plot_decision_boundary(
    rbf_svm, X_train_scaled, y_train,
    "SVM Decision Boundary - RBF Kernel"
)
```

## How to Run

1.  Install Python 3 and the required libraries.

2.  Save the program as `svm_iris.py`.

3.  Open a terminal or command prompt in the folder containing the file.

4.  Run:

    ``` bash
    python svm_iris.py
    ```

5.  Observe the accuracy values printed for both kernels and the
    decision-boundary plots.

## Expected Output

The program prints two classification accuracy values:

``` text
Linear Kernel Accuracy: <calculated accuracy> %
RBF Kernel Accuracy: <calculated accuracy> %
```

The exact values are calculated when the program runs. Two plots are
displayed, showing the decision boundaries produced by the linear and
RBF SVM models.

## Result

The Support Vector Machine classification experiment is implemented
using the Iris dataset. The performance of linear and RBF kernels is
evaluated using test-set accuracy, and their decision boundaries are
visualized for comparison.

## Notes

-   Only the first two Iris features are used so that the decision
    boundaries can be displayed in a two-dimensional plot.
-   The feature scaler is fitted only on the training data to avoid data
    leakage.
-   Accuracy depends on the selected features, train-test split, and
    model settings.
