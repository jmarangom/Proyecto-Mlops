# Machine Learning Beginner Guide Example
# Goal: Train a model to classify iris flowers

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

print("Dataset preview:")
print(X.head())

print("\nTarget classes:")
print(iris.target_names)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 8. Feature importance
feature_importance = pd.DataFrame({
    "feature": iris.feature_names,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(8, 4))
sns.barplot(
    data=feature_importance,
    x="importance",
    y="feature"
)

plt.title("Feature Importance")
plt.show()

# 9. Save the trained model
joblib.dump(model, "iris_model.pkl")

print("\nModel saved as iris_model.pkl")

# 10. Example prediction
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)
predicted_class = iris.target_names[prediction[0]]

print("\nExample Prediction:")
print(f"The predicted flower class is: {predicted_class}")