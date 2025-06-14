import streamlit as st       
import joblib
import numpy as np

# Load model
model = joblib.load("best_model/logistic_regression_best_model.pkl")

# Judul
st.title("Prediksi Kategori Obesitas")

# Form input
st.header("Masukkan Data Pengguna:")

age = st.number_input("Usia", min_value=1, max_value=100)
gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, step=0.01)
weight = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=300.0, step=0.1)
favc = st.selectbox("Sering makan makanan tinggi kalori?", ['yes', 'no'])
fcvc = st.slider("Frekuensi konsumsi sayur (1-3)", 1.0, 3.0, step=0.1)
ncp = st.slider("Jumlah makanan utama/hari", 1.0, 4.0, step=0.5)
calc = st.selectbox("Konsumsi alkohol", ['no', 'Sometimes', 'Frequently', 'Always'])
scc = st.selectbox("Apakah konsultasi dengan ahli gizi?", ['yes', 'no'])
smoke = st.selectbox("Merokok?", ['yes', 'no'])
ch2o = st.slider("Konsumsi air (liter)", 1.0, 3.0, step=0.1)
fhwo = st.selectbox("Riwayat keluarga dengan kelebihan berat badan?", ['yes', 'no'])
faf = st.slider("Frekuensi aktivitas fisik (jam/minggu)", 0.0, 4.0, step=0.5)
tue = st.slider("Waktu yang dihabiskan menggunakan teknologi (jam/hari)", 0.0, 3.0, step=0.5)
caec = st.selectbox("Frekuensi ngemil", ['no', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox("Transportasi utama", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

# Encode input
def preprocess_input():
    binary_map = {'yes': 1, 'no': 0, 'Male': 1, 'Female': 0}
    ordinal_map = {
        'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3,
        'Public_Transportation': 0, 'Walking': 1, 'Automobile': 2, 'Motorbike': 3, 'Bike': 4
    }

    input_data = [
        age,
        binary_map[gender],
        height,
        weight,
        ordinal_map[calc],
        binary_map[favc],
        fcvc,
        ncp,
        binary_map[scc],
        binary_map[smoke],
        ch2o,
        binary_map[fhwo],
        faf,
        tue,
        ordinal_map[caec],
        ordinal_map[mtrans]
    ]

    return np.array([input_data])

# Prediksi
if st.button("Prediksi"):
    input_array = preprocess_input()
    prediction = model.predict(input_array)[0]
    st.success(f"Hasil Prediksi: {prediction}")
