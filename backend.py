from flask import Flask, request, jsonify  # type: ignore[import]
from flask_cors import CORS  # type: ignore[import]  # Install this: pip install flask-cors
import numpy as np  # type: ignore[import]
import pickle
import sys
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


# Provide a compatible ConstrainedRegressor for unpickling
class ConstrainedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_model=None, scaler=None):
        self.base_model = base_model
        self.scaler = scaler

    def fit(self, X, Y):
        if hasattr(self.base_model, "fit"):
            self.base_model.fit(X, Y)
        return self

    def predict(self, X):
        preds = self.base_model.predict(X)
        try:
            X_unscaled = self.scaler.inverse_transform(X)
            Th1 = X_unscaled[:, 3]
            Tc1 = X_unscaled[:, 4]
            preds = np.asarray(preds)
            if preds.ndim == 2:
                preds[:, 0] = np.minimum(preds[:, 0], Th1 - 0.001)
                preds[:, 1] = np.clip(preds[:, 1], Tc1 + 4.0, Tc1 + 15.0)
        except Exception:
            pass
        return preds

# Ensure unpickling can find the class if pickles reference '__main__.ConstrainedRegressor'
try:
    sys.modules['__main__'].ConstrainedRegressor = ConstrainedRegressor
except Exception:
    try:
        import types
        m = types.ModuleType('__main__')
        m.ConstrainedRegressor = ConstrainedRegressor
        sys.modules['__main__'] = m
    except Exception:
        pass

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Load scaler & encoder
scaler = pickle.load(open("model/scaler.pkl", "rb"))
encoder = pickle.load(open("model/encoder.pkl", "rb"))

# Load models
models = {
    "Random Forest": pickle.load(open("model/model_RandomForest.pkl", "rb")),
    "Gradient Boosting": pickle.load(open("model/model_GradientBoosting.pkl", "rb")),
    "Linear Regression": pickle.load(open("model/model_LinearRegression.pkl", "rb")),
    "ANN": pickle.load(open("model/model_ANN.pkl", "rb")),
}

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "GET":
        return jsonify({"message": "Heat Transfer AI API is running", "status": "active"})
    
    try:
        data = request.json
        print("Received data:", data)

        model_name = data["model"]

        # Parse input values
        fluid = data["Types_of_fluid"]
        mh = float(data["mh"])
        mc = float(data["mc"])
        Th1 = float(data["Th1"])
        Tc1 = float(data["Tc1"])
        c = float(data["c"])
        h = float(data["h"])
        volume = float(data["Volume_Concentration"])
        size = float(data["Size_of_Particle"])
        ultra = float(data["Ultrasonication_time"])
        speed = float(data["Speed_of_Magnetic_Stirrer"])

        # Encode and prepare inputs
        fluid_encoded = encoder.transform([fluid])
        input_data = np.array([[ 
            fluid_encoded, mh, mc, Th1, Tc1, c, h,
            volume, size, ultra, speed
        ]])
        input_scaled = scaler.transform(input_data)

        outputs = [
            "Th2", "Tc2", "LMTD", "e", "U0", "V", "Rec",
            "Prc", "NTU", "Dec", "Nuc", "hc", "R"
        ]

        # NEW LOGIC: Check if we are comparing all models
        if model_name == "Compare All Models":
            all_predictions = {}
            for name, mod in models.items():
                prediction = mod.predict(input_scaled)
                all_predictions[name] = dict(zip(outputs, prediction.tolist()))
            
            return jsonify({"comparison": True, "results": all_predictions})
        
        # Default Logic: Single model prediction
        else:
            model = models[model_name]
            prediction = model.predict(input_scaled)
            result = dict(zip(outputs, prediction.tolist()))
            
            return jsonify({"comparison": False, "results": result})
    
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500
    
    
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "models": list(models.keys())})

if __name__ == "__main__":
    print("Starting Flask server...")
    print("Available models:", list(models.keys()))
    print("API endpoint: http://127.0.0.1:5000/predict")
    app.run(debug=True, host="127.0.0.1", port=5000)
