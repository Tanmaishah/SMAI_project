import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

with open('Prediction_data_LR/player_valuation_data.pkl', 'rb') as f:
    data = pickle.load(f)
X = data['X']
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_train)
mse = mean_squared_error(y_train, preds)
r2 = r2_score(y_train, preds)

print("\n--- Train Set Evaluation ---")

print(f"Train MSE: {mse:.3f}")
print(f"Train R^2: {r2:.3f}")

with open('Prediction_data_LR/trained_valuation_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
weights = {
    'coefficients': model.coef_.tolist(),
    'intercept': model.intercept_
}

with open('Prediction_data_LR/trained_valuation_model_weights.json', 'w') as f:
    json.dump(weights, f)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\n--- Test Set Evaluation ---")
print(f"Test MSE: {mse:.3f}")
print(f"Test R^2: {r2:.3f}")
