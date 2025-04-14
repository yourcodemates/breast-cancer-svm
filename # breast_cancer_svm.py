# breast_cancer_svm.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 1 Load the dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

# 2 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3 Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4 Train SVM model
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# 5 Make predictions
y_pred = model.predict(X_test_scaled)

# 6 Evaluate the mod
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7 Visualize accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

plt.bar(["SVM Accuracy"], [acc], color="purple")
plt.ylim(0, 1)
plt.title("Breast Cancer Prediction Accuracy")
plt.ylabel("Accuracy")
plt.savefig("sample_output.png")  # Save the figure
plt.show()
