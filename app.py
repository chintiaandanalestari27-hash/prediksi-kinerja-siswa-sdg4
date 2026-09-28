import streamlit as st
import pandas as pd
import joblib
import os

# Konfigurasi Tampilan Halaman Web
st.set_page_config(
    page_title="Prediksi Performa Siswa - SDG 4",
    page_icon="🎓",
    layout="wide"
)

# Title & Deskripsi
st.title("🎓 Aplikasi AI Prediksi Performa Siswa")
st.subheader("Pendukung SDG Goal 4: Quality Education")
st.markdown("---")

# Mengatur Pencarian Path Model (Bisa di 'model/' atau '../model/')
def get_model_path(filename):
    if os.path.exists(os.path.join("model", filename)):
        return os.path.join("model", filename)
    elif os.path.exists(os.path.join("..", "model", filename)):
        return os.path.join("..", "model", filename)
    return filename

# Sidebar: Pengaturan Model
st.sidebar.header("⚙️ Pengaturan Model AI")
pilihan_model = st.sidebar.selectbox(
    "Pilih Algoritma Model:",
    ["Random Forest", "Logistic Regression", "KNN"]
)

# Load Model Berdasarkan Pilihan
model = None
if pilihan_model == "Random Forest":
    model = joblib.load(get_model_path("random_forest_model.pkl"))
elif pilihan_model == "Logistic Regression":
    model = joblib.load(get_model_path("logreg_model.pkl"))
else:
    model = joblib.load(get_model_path("knn_model.pkl"))

st.sidebar.markdown("---")
st.sidebar.header("📋 Input Data Siswa")

# Form Input Faktor Utama Akademik & Sosial
hours_studied = st.sidebar.number_input("Jam Belajar per Minggu", min_value=0, max_value=100, value=20)
attendance = st.sidebar.slider("Tingkat Kehadiran (%)", min_value=0, max_value=100, value=85)
previous_scores = st.sidebar.number_input("Nilai Ujian Sebelumnya", min_value=0, max_value=100, value=75)
tutoring_sessions = st.sidebar.number_input("Sesi Bimbingan Belajar", min_value=0, max_value=20, value=2)
sleep_hours = st.sidebar.slider("Jam Tidur per Hari", min_value=0, max_value=12, value=7)

# Tombol Eksekusi Prediksi
if st.button("🔮 Prediksi Kelulusan Siswa", type="primary"):
    # Menyiapkan input data yang sesuai dengan fitur saat pelatihan
    input_data = pd.DataFrame({
        'Hours_Studied': [hours_studied],
        'Attendance': [attendance],
        'Parental_Involvement': [1],
        'Access_to_Resources': [1],
        'Extracurricular_Activities': [1],
        'Sleep_Hours': [sleep_hours],
        'Previous_Scores': [previous_scores],
        'Motivation_Level': [1],
        'Internet_Access': [1],
        'Tutoring_Sessions': [tutoring_sessions],
        'Family_Income': [1],
        'Teacher_Quality': [1],
        'School_Type': [1],
        'Peer_Influence': [1],
        'Physical_Activity': [2],
        'Learning_Disabilities': [0],
        'Parental_Education_Level': [1],
        'Distance_from_Home': [1],
        'Gender': [1]
    })

    # Melakukan Prediksi
    prediksi = model.predict(input_data)[0]

    st.markdown("### Hasil Analisis Sistem AI:")
    if prediksi == 1:
        st.balloons()
        st.success(f"🎉 **Status Diprediksi: LULUS**\n\n*(Model yang digunakan: {pilihan_model})*")
    else:
        st.warning(f"⚠️ **Status Diprediksi: BERISIKO TIDAK LULUS**\n\n*(Model yang digunakan: {pilihan_model})*")
        st.info("💡 **Rekomendasi Intervensi (SDG 4):** Siswa memerlukan bimbingan akademis tambahan dan peningkatan tingkat kehadiran.")