import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("heart.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
X = df.drop("target", axis=1)
y = df["target"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion matrix.png")
plt.show()
importances = model.feature_importances_

features = X.columns

plt.figure(figsize=(10,5))
plt.bar(features, importances)

plt.xticks(rotation=90)
plt.title("Feature Importance")

plt.savefig("feature importance.png")
plt.show()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))