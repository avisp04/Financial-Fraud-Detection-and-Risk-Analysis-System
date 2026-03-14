#Uncomment the Feature Engineering and data drop columns if using Baseline_dataset(extended.csv) since features are tailor made for that file
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Change "" to your target to read .CSV file
data = pd.read_csv("your/file/path/here.csv")

"""
#Feature engineering
data["TransactionHour"] = (data["TransactionDT"] % 86400) // 3600
data["DistanceRatio"] = data["dist2"] / (data["dist1"] + 1)
data["HighAmount"] = (data["TransactionAmt"] > data["TransactionAmt"].median()).astype(int)
data["CardTransactionCount"] = data.groupby("card1")["TransactionID"].transform("count")
data["CardAmountMean"] = data.groupby("card1")["TransactionAmt"].transform("mean")

data = data.drop(columns=[
    "ProductCD",
    "card4",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo"
])
"""

print("data size:", data.shape)

#Separate features and target
x = data.drop("isFraud", axis=1)
y = data["isFraud"]

#Data Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# SMOTE for class imbalance
sm = SMOTE(random_state=42)
x_train, y_train = sm.fit_resample(x_train, y_train)

#Model
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_split=4,
    class_weight="balanced",
    random_state=42
)

rf.fit(x_train, y_train)

#Probability of fraud
probs = rf.predict_proba(x_test)[:,1]

#Lowering Threshold so we get higher flags 
pred = (probs > 0.35).astype(int)

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
