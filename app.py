import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Page config
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .no-failure {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-left: 5px solid #10b981;
    }
    .failure {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-left: 5px solid #ef4444;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_components():
    """Load model and all preprocessing components"""
    try:
        # 1. Load XGBoost model
        model = joblib.load('failure_prediction_model.pkl')
        
        # 2. Load LabelEncoder
        label_encoder = joblib.load('label_encoder.pkl')
        
        # 3. Load label mapping
        with open('label_mapping.pkl', 'rb') as f:
            label_mapping = pickle.load(f)
        
        # 4. Load type mapping
        with open('type_mapping.pkl', 'rb') as f:
            type_mapping = pickle.load(f)
        
        # 5. Load features info
        with open('features_info.pkl', 'rb') as f:
            features_used = pickle.load(f)
        
        st.success("✅ Model and components loaded successfully!")
        return model, label_encoder, label_mapping, type_mapping, features_used
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Make sure all .pkl files are in the same folder as app.py")
        return None, None, None, None, None

# Load components
model, label_encoder, label_mapping, type_mapping, features_used = load_model_components()

# Failure type information
FAILURE_INFO = {
    'No Failure': {
        'description': 'The machine operates normally',
        'icon': '✅',
        'color': '#10b981',
        'actions': [
            'Continue normal operation',
            'Maintain a regular maintenance schedule',
            'Monitor key parameters'
        ],
        'severity': 'Low'
    },
    'Heat Dissipation Failure': {
        'description': 'Heat dissipation failure - cooling system problem',
        'icon': '🔥',
        'color': '#ef4444',
        'actions': [
            'Check the cooling system immediately',
            'Clean the fan and heatsink',
            'Monitor temperature closely'
        ],
        'severity': 'High'
    },
    'Power Failure': {
        'description': 'Power failure - power supply problem',
        'icon': '⚡',
        'color': '#f59e0b',
        'actions': [
            'Check power connections',
            'Check the stabilizer and UPS',
            'Verify input voltage'
        ],
        'severity': 'High'
    },
    'Overstrain Failure': {
        'description': 'Overload failure - excessive mechanical stress',
        'icon': '💪',
        'color': '#8b5cf6',
        'actions': [
            'Reduce operational expenses',
            'Check alignment and vibration',
            'Bearing and gear inspection'
        ],
        'severity': 'Medium'
    },
    'Tool Wear Failure': {
        'description': 'Tool wear failure - the cutting tool is worn out',
        'icon': '⚒️',
        'color': '#3b82f6',
        'actions': [
            'Replace worn tools',
            'Check the sharpness of the cutting edge',
            'Reset machining parameters'
        ],
        'severity': 'Medium'
    }
}

def prepare_input_data(input_dict):
    """Prepare input for prediction"""
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Rename columns to match model's expected names
        # Ubah dari 'Type' menjadi 'Type_encoded'
        input_df = input_df.rename(columns={
            'Type': 'Type_encoded'
        })

        # Map type values (ini akan ubah L/M/H ke 0/1/2)
        input_df['Type_encoded'] = input_df['Type_encoded'].map(type_mapping)
        
        # Ensure correct feature order
        input_df = input_df[features_used]
        
        return input_df
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None

def make_prediction(input_data):
    """Make predictions using the model"""
    try:
        # Prepare input
        input_df = prepare_input_data(input_data)
        if input_df is None:
            return None, None
        
        # Predict
        prediction_encoded = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Decode prediction
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        
        return prediction, prediction_proba
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ MACHINE FAILURE PREDICTION SYSTEM</h1>
        <p>Early detection of machine damage types based on Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Input Parameters
    with st.sidebar:
        st.header("⚙️ Machine Parameters")
        st.markdown("---")
        
        # Machine Type
        st.subheader("1. Machine Type")
        machine_type = st.radio(
            "Product Quality",
            options=["Low (L)", "Medium (M)", "High (H)"],
            index=1,
            help="L: 50% product, M: 30%, H: 20%"
        )
        type_code = machine_type[-2]
        
        # Sensor Parameters
        st.subheader("2. Sensor Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=295.0,
                max_value=304.5,
                value=300.0,
                step=0.1,
                help="Environmental temperature around the machine"
            )
            
            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=305.7,
                max_value=313.8,
                value=310.0,
                step=0.1,
                help="Internal temperature during operation"
            )
        
        with col2:
            rotational_speed = st.number_input(
                "Rotational Speed ​​(rpm)",
                min_value=1168,
                max_value=2886,
                value=1500,
                step=10,
                help="Machine rotation speed"
            )
            
            torque = st.number_input(
                "Torque (Nm)",
                min_value=3.8,
                max_value=76.6,
                value=40.0,
                step=0.1,
                help="The resulting rotating force"
            )
        
        tool_wear = st.slider(
            "Tool Wear (minutes)",
            min_value=0,
            max_value=253,
            value=100,
            help="Accumulated tool usage time"
        )
        
        st.markdown("---")
        
        # Predict Button
        predict_btn = st.button(
            "🚀 DAMAGE TYPE PREDICTION",
            type="primary",
            use_container_width=True
        )
        
        # Info
        with st.expander("ℹ️ Dataset Information"):
            st.write("""
            **Dataset:** Predictive Maintenance Classification
            **Samples:** 10,000
            **Failure Rate:** 3.4%
            **Model:** XGBoost Classifier
            **Accuracy:** >95%
            """)

    # Main Content
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Current Status
        st.subheader("📊 Current Machine Status")
        
        # Metrics
        cols = st.columns(4)
        metrics_data = [
            ("Machine Type", machine_type, "⚙️"),
            ("Air temperature", f"{air_temp} K", "🌡️"),
            ("Process Temperature", f"{process_temp} K", "🔥"),
            ("Speed", f"{rotational_speed} rpm", "⚡"),
            ("Torque", f"{torque} Nm", "🔧"),
            ("Tool Wear", f"{tool_wear} minute", "⏱️")
        ]
        
        for i, (label, value, icon) in enumerate(metrics_data):
            with cols[i % 4]:
                st.metric(label, value)
    
    with col_right:
        st.subheader("🎯 Info Model")
        
        if model is not None:
            st.success("The model is ready to use!")
            st.info(f"**Classes:** {len(label_encoder.classes_)} type of damage")
            st.info(f"**Features:** {len(features_used)} parameter input")
        
        # Quick Stats
        st.markdown("---")
        st.caption("**Dataset Statistics:**")
        st.caption("✅ No Failure: 96.6%")
        st.caption("⚠️ Failure Cases: 3.4%")
    
    # Input Data untuk Prediction
    input_data = {
        'Type': type_code,
        'Air temperature K': air_temp,
        'Process temperature K': process_temp,
        'Rotational speed rpm': rotational_speed,
        'Torque Nm': torque,
        'Tool wear min': tool_wear
    }
    
    # PREDICTION SECTION
    if predict_btn and model is not None:
        st.markdown("---")
        
        # Show loading
        with st.spinner("Analyze machine data..."):
            prediction, probabilities = make_prediction(input_data)
        
        if prediction is not None:
            # PREDICTION CARD
            info = FAILURE_INFO.get(prediction, {})
            is_failure = prediction != "No Failure"
            
            st.markdown(f"""
            <div class="prediction-card {'failure' if is_failure else 'no-failure'}">
                <h2>{info.get('icon', '')} PREDICTION RESULTS: {prediction}</h2>
                <p><strong>Deskripsi:</strong> {info.get('description', '')}</p>
                <p><strong>Severity Level:</strong> {info.get('severity', '')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence
            confidence = probabilities[label_encoder.transform([prediction])[0]]
            st.subheader(f"🔄 Confidence Level: {confidence:.2%}")
            st.progress(float(confidence))
            
            # REKOMENDASI
            st.subheader("📋 Recommended Action")
            
            if is_failure:
                st.warning("**⚠️ IMMEDIATE ACTION NEEDED:**")
                for i, action in enumerate(info.get('actions', []), 1):
                    st.write(f"{i}. {action}")
                
                # Emergency contact
                st.error("""
                **🆘 Emergency Contact:**
                - Technician: (021) 1234-5678
                - Supervisor: 0812-3456-7890
                - Email: maintenance@company.com
                """)
            else:
                st.success("**✅ NORMAL OPERATION:**")
                for i, action in enumerate(info.get('actions', []), 1):
                    st.write(f"{i}. {action}")
            
            # PROBABILITY DISTRIBUTION
            st.subheader("📈 Probability Distribution")
            
            # Create probability dataframe
            prob_df = pd.DataFrame({
                'Type of Damage': label_encoder.classes_,
                'Probabilitas': probabilities
            }).sort_values('Probabilitas', ascending=False)
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = [info['color'] if cls == prediction else '#cccccc' 
                     for cls in prob_df['Type of Damage']]
            
            bars = ax.barh(prob_df['Type of Damage'], 
                          prob_df['Probabilitas'], 
                          color=colors)
            ax.set_xlabel('Probabilitas', fontsize=12)
            ax.set_xlim(0, 1)
            
            # Add percentage labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{width:.1%}', va='center', fontsize=10,
                       fontweight='bold' if width == confidence else 'normal')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Data table
            st.dataframe(
                prob_df.style.format({'Probabilitas': '{:.2%}'}),
                use_container_width=True
            )
            
            # FEATURE IMPORTANCE
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                
                importance = model.feature_importances_
                feat_df = pd.DataFrame({
                    'Feature': features_used,
                    'Importance': importance
                }).sort_values('Importance', ascending=True)
                
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                bars2 = ax2.barh(feat_df['Feature'], feat_df['Importance'])
                ax2.set_xlabel('Importance Score')
                ax2.set_title('The Influence of Parameters on Prediction')
                
                # Color bars
                for bar in bars2:
                    bar.set_color('#4f46e5')
                
                plt.tight_layout()
                st.pyplot(fig2)
            
            # EXPORT REPORT
            st.subheader("📥 Export Report")
            
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            with col_exp1:
                if st.button("📋 Save as CSV"):
                    report_df = pd.DataFrame([{
                        **input_data,
                        'Prediction': prediction,
                        'Confidence': confidence,
                        'Timestamp': pd.Timestamp.now()
                    }])
                    report_df.to_csv('prediction_report.csv', index=False)
                    st.success("The report is saved!")
            
            with col_exp2:
                if st.button("🖨️ Print Summary"):
                    st.info("""
                    **SUMMARY:**
                    Prediction: {}
                    Confidence: {:.2%}
                    Time: {}
                    """.format(prediction, confidence, pd.Timestamp.now()))
    
    # ABOUT SECTION
    with st.expander("📚 About This Application"):
        st.write("""
        ### Predictive Maintenance System
        
        **Objective:**
        Predicting specific machine damage types to make maintenance processes more efficient and reduce production downtime.
        
        **Predicted Type of Damage:**
        1. **No Failure** - Machine operating normally
        2. **Heat Dissipation Failure** - Cooling system problem
        3. **Power Failure** - Power supply problem
        4. **Overstrain Failure** - Mechanical overload
        5. **Tool Wear Failure** - Cutting tool wear
        
        **Technology:**
        - Machine Learning: XGBoost Classifier
        - Framework: Scikit-learn, Streamlit
        - Deployment: Streamlit Cloud
        
        **Developed by:** Fauzan Jaelani
        **For:** Final Project Data Science
        """)

if __name__ == "__main__":
    main()