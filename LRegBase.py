import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


#Load dataset with your target
#Base Logistic Regresssion Model is for Baseline_dataset.csv
data = pd.read_csv("your/file/path/here.csv")

print("dataset shape:", data.shape)
print(data.head())


#Separating features and target
X = data.drop("isFraud", axis=1)
y = data["isFraud"]


#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Creating logistic regression model
model = LogisticRegression(max_iter=6000)

#Training the model
model.fit(X_train, y_train)


#Predictions
pred = model.predict(X_test)


#Checking accuracy
print("\naccuracy:", accuracy_score(y_test, pred))

print("\nclassification report:")
print(classification_report(y_test, pred))


#Checking fraud transactions
results = X_test.copy()
results["actual"] = y_test.values
results["predicted"] = pred

fraud = results[results["predicted"] == 1]

print("\nfraud transactions detected:")
print(fraud.head())

print("\nnumber of fraud detected:", len(fraud))
