import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict


with open('Prediction_data_LR/player_valuation_data.pkl', 'rb') as f:
    data = pickle.load(f)
X = data['X']
y = data['y']

cluster_data = defaultdict(lambda: {'X': [], 'y': []})
for features, target in zip(X, y):
        cluster = features[-1]
        cluster_data[cluster]['X'].append(features[:-1])
        cluster_data[cluster]['y'].append(target)

for cluster_id, data in cluster_data.items():

    print(f"\n--- Cluster {cluster_id} --- \n")

    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y'], test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_train)
    mse = mean_squared_error(y_train, preds)
    r2 = r2_score(y_train, preds)

    print("--- Train Set Evaluation ---")
    print(f"Train MSE: {mse:.3f}")
    print(f"Train R^2: {r2:.3f}")

    with open(f'Prediction_data_LR/trained_valuation_model_{cluster_id}.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    weights = {
        'coefficients': model.coef_.tolist(),
        'intercept': model.intercept_
    }

    with open(f'Prediction_data_LR/trained_valuation_model_weights_{cluster_id}.json', 'w') as f:
        json.dump(weights, f)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("--- Test Set Evaluation ---")
    print(f"Test MSE: {mse:.3f}")
    print(f"Test R^2: {r2:.3f}")
