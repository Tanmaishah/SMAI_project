import pickle
import os

def load_model():
    model_path = 'trained_valuation_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_market_value(previous_valuation, score, age, cluster):
    """
    Inputs:
    - previous_valuation: float (in millions)
    - score: float
    - age: int
    - cluster: int
    """
    model = load_model()
    input_features = [[previous_valuation, score, age, cluster]]
    prediction = model.predict(input_features)[0]
    return prediction

if __name__ == '__main__':
    try:
        print("Enter player data for market value prediction:")
        previous_valuation = float(input("Previous market value (in millions): "))
        score = float(input("Score (1 to 100): "))
        age = int(input("Age: "))
        cluster = int(input("Cluster ID (1 to 8): "))

        predicted_value = predict_market_value(previous_valuation, score, age, cluster)
        print(f"Predicted future market value: €{predicted_value:.2f}M")
    except Exception as e:
        print(f"Error: {e}")
