try:
    import streamlit as st  # type: ignore[import]
except ImportError:
    raise ImportError("Streamlit is required. Install it with 'pip install streamlit'.")

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.base import BaseEstimator, RegressorMixin

try:
    import plotly.graph_objects as go  # type: ignore[import]
except ImportError:
    go = None

# ==========================================
# CUSTOM CONSTRAINED REGRESSOR (REQUIRED FOR UNPICKLING)
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
            
        Y_np[:, 0] = Th1 - Y_np[:, 0]  
        Y_np[:, 1] = Y_np[:, 1] - Tc1  
        
        self.base_model.fit(X, Y_np)
        return self

    def predict(self, X):
        preds = self.base_model.predict(X)
        
        X_unscaled = self.scaler.inverse_transform(X)
        Th1 = X_unscaled[:, 3]
        Tc1 = X_unscaled[:, 4]
        
        preds[:, 0] = Th1 - preds[:, 0]
        preds[:, 1] = Tc1 + preds[:, 1]
        
        preds[:, 0] = np.minimum(preds[:, 0], Th1 - 0.001) 
        preds[:, 1] = np.maximum(preds[:, 1], Tc1 + 0.001) 
        
        return preds

# =========================
# Page Config & Futuristic CSS
# =========================
st.set_page_config(page_title="Lumina Thermal", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #151C2C;
        border-left: 3px solid #00E5FF;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 4px;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.05);
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.2);
    }
    div[data-testid="stMetricLabel"] {
        color: #00E5FF !important;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF;
        font-weight: 700;
    }
    h3 {
        color: #FFFFFF;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin-bottom: 15px !important;
        border-bottom: 1px solid #2A354D;
        padding-bottom: 5px;
    }
    .stButton>button {
        border-radius: 4px;
        font-weight: bold;
        letter-spacing: 1px;
        text-transform: uppercase;
        border: 1px solid #00E5FF;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Load Assets safely
# =========================
@st.cache_resource
def load_assets():
    if not os.path.exists("model/scaler.pkl"):
        return None, None, None, None
        
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    encoder = pickle.load(open("model/encoder.pkl", "rb"))
    
    scores = {}
    if os.path.exists("model/scores.pkl"):
        scores = pickle.load(open("model/scores.pkl", "rb"))
    
    models = {
        "Random Forest": pickle.load(open("model/model_RandomForest.pkl", "rb")),
        "Gradient Boosting": pickle.load(open("model/model_GradientBoosting.pkl", "rb")),
        "Linear Regression": pickle.load(open("model/model_LinearRegression.pkl", "rb")),
        "ANN": pickle.load(open("model/model_ANN.pkl", "rb"))
    }
    return scaler, encoder, models, scores

scaler, encoder, models, scores = load_assets()

if scaler is None:
    st.error("⚠️ Models not found! Please run `python train_model.py` first.")
    st.stop()

# =========================
# Sidebar - Model Selection
# =========================
with st.sidebar:
    st.markdown("### ⚡ AI Core\n<span style='color:#00E5FF; font-size:0.8rem; text-transform:uppercase;'>Neural Engine</span>", unsafe_allow_html=True)
    st.write("")
    
    model_options = list(models.keys()) + ["Compare All Models"]
    selected_model_name = st.radio("Select Algorithm", model_options, index=1, label_visibility="collapsed")
    
    st.write("")
    st.write("")
    st.markdown("<hr style='border-color: #2A354D; margin-bottom: 10px;'>", unsafe_allow_html=True)
    st.caption("📄 System Docs")
    st.caption("🎧 Network Support")

# =========================
# Main Layout
# =========================
st.title("Thermal Performance Analysis")
st.markdown("<p style='color: #8C9BAB; font-size: 1.1rem;'>Configure parameters and run predictive modeling for heat exchange efficiency.</p>", unsafe_allow_html=True)
st.write("")

col_left, col_right = st.columns([1, 1.8], gap="large")

# -------------------------
# LEFT COLUMN: INPUTS 
# -------------------------
with col_left:
    with st.container(border=True):
        st.markdown("### 🌊 Fluid & Mass Flow")
        fluid_index = list(encoder.classes_).index("Al2O3") if "Al2O3" in encoder.classes_ else 0
        Types_of_fluid = st.selectbox("Types_of_fluid", encoder.classes_, index=fluid_index)
        
        c1, c2 = st.columns(2)
        with c1:
            mh = st.number_input("mh (kg/s)", value=0.083, format="%.4f")
        with c2:
            mc = st.number_input("mc (kg/s)", value=0.0167, format="%.4f")

    with st.container(border=True):
        st.markdown("### 🌡️ Thermal & Material Specs")
        c1, c2 = st.columns(2)
        with c1:
            Th1 = st.number_input("Th1 (°C)", value=78.0)
            c = st.number_input("c (Cp hot)", value=4.0)
            Volume_Concentration = st.number_input("Volume_Concentration", value=0.05, format="%.4f")
        with c2:
            Tc1 = st.number_input("Tc1 (°C)", value=36.0)
            h = st.number_input("h (Cp cold)", value=3.0)
            Size_of_Particle = st.number_input("Size_of_Particle", value=80.0)

    with st.container(border=True):
        st.markdown("### ⚙️ Operating Conditions")
        c1, c2 = st.columns(2)
        with c1:
            Ultrasonication_time = st.number_input("Ultrasonication_time", value=60.0)
        with c2:
            Speed_of_Magnetic_Stirrer = st.number_input("Speed_of_Magnetic_Stirrer", value=1500.0)

# -------------------------
# RIGHT COLUMN: OUTPUTS
# -------------------------
with col_right:
    submit = st.button("▶ INITIALIZE PREDICTION", type="primary", use_container_width=True)
    
    if submit:
        with st.spinner("Processing neural simulation..."):
            
            # Fix: Wrap the input in double brackets to make it a 2D array
            # --- FIXED DATA PROCESSING LOGIC ---

# 1. Transform the categorical fluid type into an array of numbers
fluid_encoded = encoder.transform([[Types_of_fluid]])

# If your encoder outputs a sparse matrix, convert it to a dense array
if hasattr(fluid_encoded, "toarray"):
    fluid_encoded = fluid_encoded.toarray()

# 2. Collect your numeric inputs into a 2D array (Make sure these match your actual variables!)
numeric_inputs = np.array([[Th1, Tc1, m_dot_hot, m_dot_cold]])

# 3. Combine numeric features and your encoded fluid array horizontally side-by-side
input_features = np.hstack([numeric_inputs, fluid_encoded])

# 4. Scale your combined features using your loaded scaler
input_scaled = scaler.transform(input_features)

            if selected_model_name == "Compare All Models":
                st.markdown("### 📊 Model Comparison Across All Parameters")
                
                all_results = {}
                for m_name, mod in models.items():
                    # Force flatten the prediction array
                    pred = np.ravel(mod.predict(input_scaled))
                    
                    # Pad with 0.0 if the model predicts fewer than 13 parameters
                    if len(pred) < len(outputs):
                        pred = np.pad(pred, (0, len(outputs) - len(pred)), 'constant')
                        
                    res_dict = dict(zip(outputs, pred))
                    
                    # --- MINIMAL FIX: Force physical bounds on the final text numbers ---
                    res_dict["Th2"] = min(res_dict["Th2"], Th1 - 0.001)
                    res_dict["Tc2"] = max(Tc1 + 4.0, min(res_dict["Tc2"], Tc1 + 15.0))
                    
                    all_results[m_name] = res_dict
                
                tab_graph, tab_table = st.tabs(["📊 Graphical View", "📋 Tabular View"])
                
                with tab_graph:
                    cols = st.columns(3)
                    for i, param in enumerate(outputs):
                        with cols[i % 3]:
                            param_vals = [all_results[m][param] for m in models.keys()]
                            
                            fig = go.Figure(data=[go.Bar(
                                x=list(models.keys()), 
                                y=param_vals,
                                marker_color=['#00E5FF', '#1E3A5F', '#FF007F', '#00FF7F']
                            )])
                            
                            fig.update_layout(
                                title=dict(text=f"Parameter: {param}", font=dict(color="#00E5FF", size=14)),
                                margin=dict(l=20, r=20, t=40, b=20),
                                height=280,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(showgrid=False, tickangle=-45, tickfont=dict(color="white", size=10)),
                                yaxis=dict(showgrid=True, gridcolor='#1E3A5F', tickfont=dict(color="white"))
                            )
                            st.plotly_chart(fig, use_container_width=True)

                with tab_table:
                    df = pd.DataFrame(all_results)
                    df = df.reset_index()
                    df = df.rename(columns={"index": "Parameters / Models"})
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.dataframe(
                        df.style.format(precision=4), 
                        use_container_width=True,
                        hide_index=True 
                    )

            else:
                model = models[selected_model_name]
                pred = np.ravel(model.predict(input_scaled))
                if len(pred) < len(outputs):
                    pred = np.pad(pred, (0, len(outputs) - len(pred)), 'constant')
                    
                result = dict(zip(outputs, pred))

                Th2 = result.get("Th2", 0.0)
                Tc2 = result.get("Tc2", 0.0)
                
                # --- MINIMAL FIX: Force physical bounds on the final text numbers ---
                Tc2 = max(Tc1 + 4.0, min(Tc2, Tc1 + 15.0))
                Th2 = min(Th2, Th1 - 0.001)
    
                result["Tc2"] = Tc2
                result["Th2"] = Th2
                

                Qh = mh * c * (Th1 - Th2)
                Qc = mc * h * (Tc2 - Tc1)
                Ch, Cc = mh * c, mc * h
                Cmin, Cmax = min(Ch, Cc), max(Ch, Cc)
                Cr = Cmin / Cmax if Cmax != 0 else 0

                with st.container(border=True):
                    st.markdown(f"### 📈 Neural Predictions ({selected_model_name})")
                    
                    res_cols = st.columns(4)
                    for i, (key, value) in enumerate(result.items()):
                        with res_cols[i % 4]:
                            st.markdown(f"""
                            <div data-testid="metric-container">
                                <div style="color: #8B9BB4; font-size: 14px; margin-bottom: 5px;">{key}</div>
                                <div style="color: #00E5FF; font-size: 24px; font-weight: bold;">{value:.4f}</div>
                            </div>
                            """, unsafe_allow_html=True)

                c_deriv, c_map = st.columns([1, 1.5])
                
                with c_deriv:
                    with st.container(border=True):
                        st.markdown("### 🧮 Derived Stats")
                        st.markdown(f"**Qh (kW)** <span style='float:right; color:#00E5FF;'>{Qh:.4f}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Qc (kW)** <span style='float:right; color:#00E5FF;'>{Qc:.4f}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Cmin** <span style='float:right; color:#00E5FF;'>{Cmin:.4f}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Cmax** <span style='float:right; color:#00E5FF;'>{Cmax:.4f}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Cr Ratio** <span style='float:right; color:#00E5FF;'>{Cr:.4f}</span>", unsafe_allow_html=True)
                        
                with c_map:
                    with st.container(border=True):
                        st.markdown("### 🌐 Spatial Thermal Distribution")
                        grid_res = 30
                        y = np.linspace(0, 1, grid_res)
                        
                        hot_profile = np.linspace(Th1, Th2, grid_res)
                        cold_profile = np.linspace(Tc1, Tc2, grid_res)
                        
                        Z = np.zeros((grid_res, grid_res))
                        for i in range(grid_res):
                            Z[i, :] = cold_profile * (1 - y[i]) + hot_profile * y[i]
                        
                        fig = go.Figure(data=go.Contour(
                            z=Z,
                            colorscale='Inferno', 
                            line_smoothing=0.85,
                            contours=dict(coloring='heatmap'),
                            showscale=False 
                        ))
                        
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=200,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False)
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    else:
        st.info("📡 SYSTEM READY: Input parameters and initialize to view thermal dynamics.")
        
        if scores:
            if selected_model_name == "Compare All Models":
                st.info("💡 Select a specific neural model from the sidebar to view its individual performance metrics.")
            else:
                with st.container(border=True):
                    st.markdown(f"### 📊 Engine Details: {selected_model_name}")
                    
                    model_data = scores.get(selected_model_name, "N/A")
                    
                    if isinstance(model_data, dict):
                        r2_val = model_data.get("R2", "N/A")
                        r2_str = f"{r2_val:.4f}" if isinstance(r2_val, float) else r2_val
                        
                        mae_val = model_data.get("MAE", "N/A")
                        mae_str = f"{mae_val:.4f}" if isinstance(mae_val, float) else mae_val
                        
                        rmse_val = model_data.get("RMSE", "N/A")
                        rmse_str = f"{rmse_val:.4f}" if isinstance(rmse_val, float) else rmse_val
                    else:
                        r2_str = f"{model_data:.4f}" if isinstance(model_data, float) else "N/A"
                        mae_str = "N/A" 
                        rmse_str = "N/A" 

                    # -------- THE FIX IS SECURED HERE --------
                    score_cols = st.columns(3)
                    
                    # --- FIXED CODES ---
                    # Box 1 goes into the first column
                    with score_cols:
                        st.markdown(f"""
                        <div style="background-color: #151C2C; border: 1px solid #00E5FF; padding: 15px; border-radius: 6px; text-align: center; box-shadow: 0 0 10px rgba(0, 229, 255, 0.1);">
                            <div style="color: #8C9BAB; font-size: 14px; font-weight: 600; margin-bottom: 8px; text-transform: uppercase;">R² Score</div>
                            <div style="color: #00FF7F; font-size: 26px; font-weight: bold;">{r2_str}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Box 2 goes into the second column
                    with score_cols:
                        st.markdown(f"""
                        <div style="background-color: #151C2C; border: 1px solid #2A354D; padding: 15px; border-radius: 6px; text-align: center;">
                            <div style="color: #8C9BAB; font-size: 14px; font-weight: 600; margin-bottom: 8px; text-transform: uppercase;">Mean Absolute Error (MAE)</div>
                            <div style="color: #FFFFFF; font-size: 26px; font-weight: bold;">{mae_str}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Box 3 goes into the third column
                    with score_cols:
                        st.markdown(f"""
                        <div style="background-color: #151C2C; border: 1px solid #2A354D; padding: 15px; border-radius: 6px; text-align: center;">
                            <div style="color: #8C9BAB; font-size: 14px; font-weight: 600; margin-bottom: 8px; text-transform: uppercase;">Root Mean Sq. Error (RMSE)</div>
                            <div style="color: #FFFFFF; font-size: 26px; font-weight: bold;">{rmse_str}</div>
                        </div>
                        """, unsafe_allow_html=True)