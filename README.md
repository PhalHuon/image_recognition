# Machine Learning Image Recognition: PCA & Digit Classification

This project demonstrates the use of **Principal Component Analysis (PCA)** and **supervised learning** for recognizing **handwritten digits**. We use an image dataset where each digit is represented as an 8x8 grayscale image, flattened into a 64-dimensional feature vector. 

Because visualizing data in 64 dimensions isn't humanly possible, we apply PCA to reduce the dimensionality to **2D**, enabling us to explore the structure and clustering of the dataset visually.

---

## Dataset

We use the classic `digits` dataset from `scikit-learn`. Each sample is:
- A grayscale image of size `8x8` pixels
- Flattened into a 64-element feature vector
- Labeled with a digit (0–9)

---

## Objective

- **Visualize** high-dimensional data by projecting it into **2D** using PCA.
- Use **machine learning algorithms** (e.g., KMeans or a classifier) to interpret and classify the digits.
- **Understand patterns** and **clusters** in the digit space.

---

## PCA Visualization

Since we can't visualize all 64 features simultaneously, **Principal Component Analysis** helps us:

- Reduce the dimensionality from 64 ➝ 2
- Plot the data in a **2D scatter plot**
- Reveal natural groupings and distributions of digits

---

## Example Visualization

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
digits = load_digits()
X = digits.data
y = digits.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.title("PCA of Handwritten Digits")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()
