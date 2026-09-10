import joblib

model = joblib.load("model/fraud_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

importances = model.feature_importances_

paired = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)

print("\nFeature importance (highest = most influence on prediction):\n")
for name, score in paired:
    bar = "#" * int(score * 100)
    print(f"{name:25s} {score:.4f}  {bar}")