import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Load Dataset with target 
#Base Random Forest model to work with Baseline_dataset
data = pd.read_csv("D:/Coding/Python/fraud_detection/IEEE_dataset.csv")

print("data size:", data.shape)

#Separate features and target
x = data.drop("isFraud", axis=1)
y = data["isFraud"]

#Data Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#Model
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
rf.fit(x_train, y_train)

#Probability of fraud
probs = rf.predict_proba(x_test)[:,1]

#Lowering Threshold so we get higher flags 
pred = (probs > 0.3).astype(int)

# accuracy
acc = accuracy_score(y_test, pred)

print("\naccuracy:", acc)

print("\nreport:\n")
print(classification_report(y_test, pred))

#Displaying Fraud Transactions

results = x_test.copy()
results["ActualFraud"] = y_test.values
results["PredictedFraud"] = pred

fraud_transactions = results[results["PredictedFraud"] == 1]

print("\nFraud transactions detected by model:\n")
print(fraud_transactions.head(10))

print("\nTotal fraud detected:", fraud_transactions.shape[0])
