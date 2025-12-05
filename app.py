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
    """Load model dan semua preprocessing components"""
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
        
        st.success("✅ Model dan components berhasil diload!")
        return model, label_encoder, label_mapping, type_mapping, features_used
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Pastikan semua file .pkl ada di folder yang sama dengan app.py")
        return None, None, None, None, None

# Load components
model, label_encoder, label_mapping, type_mapping, features_used = load_model_components()

# Failure type information
FAILURE_INFO = {
    'No Failure': {
        'description': 'Mesin beroperasi normal',
        'icon': '✅',
        'color': '#10b981',
        'actions': [
            'Lanjutkan operasi normal',
            'Pertahankan jadwal maintenance rutin',
            'Pantau parameter kunci'
        ],
        'severity': 'Low'
    },
    'Heat Dissipation Failure': {
        'description': 'Gagal disipasi panas - sistem pendingin bermasalah',
        'icon': '🔥',
        'color': '#ef4444',
        'actions': [
            'Cek sistem pendingin segera',
            'Bersihkan kipas dan heatsink',
            'Pantau suhu secara ketat'
        ],
        'severity': 'High'
    },
    'Power Failure': {
        'description': 'Gagal daya - masalah suplai listrik',
        'icon': '⚡',
        'color': '#f59e0b',
        'actions': [
            'Periksa koneksi daya',
            'Cek stabilizer dan UPS',
            'Verifikasi tegangan input'
        ],
        'severity': 'High'
    },
    'Overstrain Failure': {
        'description': 'Gagal kelebihan beban - stres mekanis berlebihan',
        'icon': '💪',
        'color': '#8b5cf6',
        'actions': [
            'Kurangi beban operasional',
            'Cek alignment dan vibration',
            'Inspeksi bearing dan gear'
        ],
        'severity': 'Medium'
    },
    'Tool Wear Failure': {
        'description': 'Gagal keausan tool - alat potong sudah aus',
        'icon': '⚒️',
        'color': '#3b82f6',
        'actions': [
            'Ganti tool yang aus',
            'Cek ketajaman cutting edge',
            'Atur ulang parameter machining'
        ],
        'severity': 'Medium'
    }
}

def prepare_input_data(input_dict):
    """Prepare input untuk prediction"""
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
    """Lakukan prediksi menggunakan model"""
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
        <h1>⚙️ SISTEM PREDIKSI KERUSAKAN MESIN</h1>
        <p>Deteksi dini jenis kerusakan mesin berbasis Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Input Parameters
    with st.sidebar:
        st.header("⚙️ Parameter Mesin")
        st.markdown("---")
        
        # Machine Type
        st.subheader("1. Tipe Mesin")
        machine_type = st.radio(
            "Kualitas Produk",
            options=["Rendah (L)", "Sedang (M)", "Tinggi (H)"],
            index=1,
            help="L: 50% produk, M: 30%, H: 20%"
        )
        type_code = machine_type[-2]
        
        # Sensor Parameters
        st.subheader("2. Parameter Sensor")
        
        col1, col2 = st.columns(2)
        with col1:
            air_temp = st.number_input(
                "Suhu Udara (K)",
                min_value=295.0,
                max_value=304.5,
                value=300.0,
                step=0.1,
                help="Suhu lingkungan sekitar mesin"
            )
            
            process_temp = st.number_input(
                "Suhu Proses (K)",
                min_value=305.7,
                max_value=313.8,
                value=310.0,
                step=0.1,
                help="Suhu internal selama operasi"
            )
        
        with col2:
            rotational_speed = st.number_input(
                "Kecepatan Rotasi (rpm)",
                min_value=1168,
                max_value=2886,
                value=1500,
                step=10,
                help="Kecepatan putar mesin"
            )
            
            torque = st.number_input(
                "Torsi (Nm)",
                min_value=3.8,
                max_value=76.6,
                value=40.0,
                step=0.1,
                help="Gaya putar yang dihasilkan"
            )
        
        tool_wear = st.slider(
            "Keausan Tool (menit)",
            min_value=0,
            max_value=253,
            value=100,
            help="Akumulasi waktu pakai tool"
        )
        
        st.markdown("---")
        
        # Predict Button
        predict_btn = st.button(
            "🚀 PREDIKSI JENIS KERUSAKAN",
            type="primary",
            use_container_width=True
        )
        
        # Info
        with st.expander("ℹ️ Informasi Dataset"):
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
        st.subheader("📊 Status Mesin Saat Ini")
        
        # Metrics
        cols = st.columns(4)
        metrics_data = [
            ("Tipe Mesin", machine_type, "⚙️"),
            ("Suhu Udara", f"{air_temp} K", "🌡️"),
            ("Suhu Proses", f"{process_temp} K", "🔥"),
            ("Kecepatan", f"{rotational_speed} rpm", "⚡"),
            ("Torsi", f"{torque} Nm", "🔧"),
            ("Tool Wear", f"{tool_wear} menit", "⏱️")
        ]
        
        for i, (label, value, icon) in enumerate(metrics_data):
            with cols[i % 4]:
                st.metric(label, value)
    
    with col_right:
        st.subheader("🎯 Model Info")
        
        if model is not None:
            st.success("Model siap digunakan!")
            st.info(f"**Classes:** {len(label_encoder.classes_)} jenis kerusakan")
            st.info(f"**Features:** {len(features_used)} parameter input")
        
        # Quick Stats
        st.markdown("---")
        st.caption("**Statistik Dataset:**")
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
        with st.spinner("Menganalisis data mesin..."):
            prediction, probabilities = make_prediction(input_data)
        
        if prediction is not None:
            # PREDICTION CARD
            info = FAILURE_INFO.get(prediction, {})
            is_failure = prediction != "No Failure"
            
            st.markdown(f"""
            <div class="prediction-card {'failure' if is_failure else 'no-failure'}">
                <h2>{info.get('icon', '')} HASIL PREDIKSI: {prediction}</h2>
                <p><strong>Deskripsi:</strong> {info.get('description', '')}</p>
                <p><strong>Tingkat Keparahan:</strong> {info.get('severity', '')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence
            confidence = probabilities[label_encoder.transform([prediction])[0]]
            st.subheader(f"🔄 Tingkat Keyakinan: {confidence:.2%}")
            st.progress(float(confidence))
            
            # REKOMENDASI
            st.subheader("📋 Rekomendasi Tindakan")
            
            if is_failure:
                st.warning("**⚠️ PERLU TINDAKAN SEGERA:**")
                for i, action in enumerate(info.get('actions', []), 1):
                    st.write(f"{i}. {action}")
                
                # Emergency contact
                st.error("""
                **🆘 Kontak Emergency:**
                - Teknisi: (021) 1234-5678
                - Supervisor: 0812-3456-7890
                - Email: maintenance@company.com
                """)
            else:
                st.success("**✅ OPERASI NORMAL:**")
                for i, action in enumerate(info.get('actions', []), 1):
                    st.write(f"{i}. {action}")
            
            # PROBABILITY DISTRIBUTION
            st.subheader("📈 Distribusi Probabilitas")
            
            # Create probability dataframe
            prob_df = pd.DataFrame({
                'Jenis Kerusakan': label_encoder.classes_,
                'Probabilitas': probabilities
            }).sort_values('Probabilitas', ascending=False)
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = [info['color'] if cls == prediction else '#cccccc' 
                     for cls in prob_df['Jenis Kerusakan']]
            
            bars = ax.barh(prob_df['Jenis Kerusakan'], 
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
                ax2.set_title('Pengaruh Parameter terhadap Prediksi')
                
                # Color bars
                for bar in bars2:
                    bar.set_color('#4f46e5')
                
                plt.tight_layout()
                st.pyplot(fig2)
            
            # EXPORT REPORT
            st.subheader("📥 Export Laporan")
            
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            with col_exp1:
                if st.button("📋 Simpan sebagai CSV"):
                    report_df = pd.DataFrame([{
                        **input_data,
                        'Prediction': prediction,
                        'Confidence': confidence,
                        'Timestamp': pd.Timestamp.now()
                    }])
                    report_df.to_csv('prediction_report.csv', index=False)
                    st.success("Laporan disimpan!")
            
            with col_exp2:
                if st.button("🖨️ Print Summary"):
                    st.info("""
                    **SUMMARY:**
                    Prediksi: {}
                    Confidence: {:.2%}
                    Waktu: {}
                    """.format(prediction, confidence, pd.Timestamp.now()))
    
    # ABOUT SECTION
    with st.expander("📚 Tentang Aplikasi Ini"):
        st.write("""
        ### Predictive Maintenance System
        
        **Tujuan:**
        Memprediksi jenis kerusakan mesin spesifik untuk membuat proses maintenance lebih efisien dan mengurangi downtime produksi.
        
        **Jenis Kerusakan yang Diprediksi:**
        1. **No Failure** - Mesin beroperasi normal
        2. **Heat Dissipation Failure** - Masalah sistem pendingin
        3. **Power Failure** - Masalah suplai daya listrik
        4. **Overstrain Failure** - Kelebihan beban mekanis
        5. **Tool Wear Failure** - Keausan alat potong
        
        **Teknologi:**
        - Machine Learning: XGBoost Classifier
        - Framework: Scikit-learn, Streamlit
        - Deployment: Streamlit Cloud
        
        **Developed by:** Fauzan Jaelani
        **Untuk:** Final Project Data Science
        """)

if __name__ == "__main__":
    main()