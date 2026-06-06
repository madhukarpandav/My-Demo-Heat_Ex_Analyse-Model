import pandas as pd
import numpy as np
import pickle
import os
from scipy import stats

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# ==========================================
# CUSTOM CONSTRAINED REGRESSOR
# ==========================================
class ConstrainedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_model, scaler):
        self.base_model = base_model
        self.scaler = scaler

    def fit(self, X, Y):
        X_unscaled = self.scaler.inverse_transform(X)
        Th1 = X_unscaled[:, 3] 
        Tc1 = X_unscaled[:, 4] 
        
        if isinstance(Y, pd.DataFrame):
            Y_np = Y.values.copy()
        else:
            Y_np = np.array(Y).copy()
            
        # Transform absolute temperatures into temperature differences (Delta T)
        Y_np[:, 0] = Th1 - Y_np[:, 0]  
        Y_np[:, 1] = Y_np[:, 1] - Tc1  
        
        # --- FIX 1: BOUND TRAINING DATA ---
        # Force the algorithm to only learn Delta Tc values between 4°C and 15°C
        # This prevents the model from being confused by outliers in data.csv
        Y_np[:, 1] = np.clip(Y_np[:, 1], 4.0, 15.0)
        
        self.base_model.fit(X, Y_np)
        return self

    def predict(self, X):
        preds = self.base_model.predict(X)
        
        X_unscaled = self.scaler.inverse_transform(X)
        Th1 = X_unscaled[:, 3]
        Tc1 = X_unscaled[:, 4]
        
        # Reconstruct absolute temperatures from predicted Delta T
        preds[:, 0] = Th1 - preds[:, 0]
        preds[:, 1] = Tc1 + preds[:, 1]
        
        # --- FIX 2: BOUND PREDICTIONS ---
        # Physical constraint: Th2 must be lower than Th1
        preds[:, 0] = np.minimum(preds[:, 0], Th1 - 0.001) 
        
        # Physical constraint: Tc2 must be between (Tc1 + 4) and (Tc1 + 15)
        preds[:, 1] = np.clip(preds[:, 1], Tc1 + 4.0, Tc1 + 15.0) 
        
        return preds


# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

print("Loading Data...")
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

X_cols = [
    "Types_of_fluid", "mh", "mc", "Th1", "Tc1", "c", "h",
    "Volume_Concentration", "Size_of_Particle",
    "Ultrasonication_time", "Speed_of_Magnetic_Stirrer"
]

Y_cols = [
    "Th2", "Tc2", "LMTD", "e", "U0", "V", "Rec",
    "Prc", "NTU", "Dec", "Nuc", "hc", "R"
]

# ==========================================
# AUTO-CLEANING: OUTLIER REMOVAL
# ==========================================
print(f"Original dataset size: {len(df)} rows")
numeric_y = df[Y_cols].select_dtypes(include=[np.number])
z_scores = np.abs(stats.zscore(numeric_y))
df_clean = df[(z_scores < 3).all(axis=1)].copy()
print(f"Cleaned dataset size: {len(df_clean)} rows")

X = df_clean[X_cols].copy()
Y = df_clean[Y_cols].copy()

# Encode & Scale
encoder = LabelEncoder()
X["Types_of_fluid"] = encoder.fit_transform(X["Types_of_fluid"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

# ==========================================
# ADVANCED MODEL DEFINITIONS
# ==========================================
models = {
    "Random Forest": ConstrainedRegressor(
        MultiOutputRegressor(RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=42, n_jobs=-1)),
        scaler
    ),
    "Gradient Boosting": ConstrainedRegressor(
        MultiOutputRegressor(HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400, random_state=42)),
        scaler
    ),
    "Linear Regression": ConstrainedRegressor(
        MultiOutputRegressor(Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('ridge', Ridge(alpha=0.5)) 
        ])),
        scaler
    ),
    "ANN": ConstrainedRegressor(
        MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(100, 50), activation="relu", solver="lbfgs",
                                          max_iter=6000, alpha=0.1, random_state=42)),
        scaler
    )
}

scores = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    
    score = r2_score(Y_test, preds, multioutput="uniform_average")
    scores[name] = score
    print(f"✅ {name} R² Score: {score:.4f}")
    
    safe_name = name.replace(" ", "")
    pickle.dump(model, open(f"model/model_{safe_name}.pkl", "wb"))

# Save Scaler, Encoder, and Scores
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(encoder, open("model/encoder.pkl", "wb"))
pickle.dump(scores, open("model/scores.pkl", "wb"))

print("\n🚀 All optimized models trained and saved!")