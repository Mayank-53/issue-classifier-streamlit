import joblib

model = joblib.load("model.pkl")

test_cases = [
    "My laptop overheats and turns off",
    "My keyboard is not working",
    "Battery drains too fast",
    "WiFi keeps disconnecting",
    "System is very slow"
]

for issue in test_cases:
    predicted = model.predict([issue])[0]
    print(f"Issue: {issue}\nPredicted Category: {predicted}\n")
