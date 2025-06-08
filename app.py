import streamlit as st
import pandas as pd
import numpy as np
import joblib # Sử dụng joblib vì bạn đã lưu mô hình bằng joblib

# --- 1. Load mô hình đã huấn luyện ---
# Đảm bảo đường dẫn đến tệp mô hình là chính xác và tệp nằm cùng thư mục với app.py
try:
    model = joblib.load('random_forest_diabetes_model.joblib')
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy tệp 'random_forest_diabetes_model.joblib'. Hãy đảm bảo tệp mô hình nằm cùng thư mục với app.py")
    st.stop() # Dừng ứng dụng nếu không tìm thấy mô hình

# --- 2. Cài đặt tiêu đề và mô tả ứng dụng ---
st.set_page_config(page_title="Ứng dụng Dự đoán Nguy cơ mắc bệnh Tiểu đường", layout="centered")
# Thay đổi tiêu đề chính theo yêu cầu
st.title("Dự đoán Nguy cơ mắc bệnh Tiểu đường")
st.write("Ứng dụng này sử dụng mô hình Machine Learning để dự đoán nguy cơ mắc bệnh tiểu đường dựa trên các thông số đầu vào.")
st.write("---")

# --- 3. Thu thập dữ liệu đầu vào từ người dùng ---
st.header("Nhập các thông số của bệnh nhân:")

age = st.slider("Tuổi", 1, 110, 30)

sex_option = st.radio("Giới tính", ["Nam", "Nữ"])
sex_encoded = 0 if sex_option == "Nam" else 1

bmi = st.slider("Chỉ số BMI", 10.0, 70.0, 25.0, format="%.1f")
waist_circumference = st.slider("Vòng eo (cm)", 40, 150, 90) # Phạm vi hợp lý cho vòng eo

fasting_blood_glucose = st.slider("Đường huyết lúc đói (mg/dL)", 0, 300, 100)
hba1c = st.slider("Chỉ số HbA1c (%)", 0.0, 15.0, 5.7, format="%.1f")

family_history_option = st.radio("Có tiền sử gia đình mắc bệnh tiểu đường không?", ["Không", "Có"])
family_history_encoded = 1 if family_history_option == "Có" else 0

gestational_diabetes_option = st.radio("Có tiền sử tiểu đường thai kỳ trước đây không?", ["Không", "Có"])
gestational_diabetes_encoded = 1 if gestational_diabetes_option == "Có" else 0

# GOM DỮ LIỆU VÀO DATAFRAME - ĐÂY LÀ ĐOẠN CẦN CHÍNH XÁC NHẤT!
# DANH SÁCH TÊN CỘT TRONG `columns` PHẢI KHỚP ĐÚNG THỨ TỰ VÀ TÊN VỚI LÚC HUẤN LUYỆN MÔ HÌNH.
input_data = pd.DataFrame([[
    age,
    sex_encoded,
    bmi,
    waist_circumference,
    fasting_blood_glucose,
    hba1c,
    family_history_encoded,
    gestational_diabetes_encoded
]],
columns=[
    'Age',
    'Sex',
    'BMI',
    'Waist_Circumference',
    'Fasting_Blood_Glucose',
    'HbA1c',
    'Family_History_of_Diabetes',
    'Previous_Gestational_Diabetes'
])

# --- 4. Nút dự đoán và hiển thị kết quả ---
st.write("---")
if st.button("Dự đoán"):
    # Thực hiện dự đoán
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data) # Lấy xác suất của cả 2 lớp

    st.subheader("Kết quả dự đoán:")
    if prediction[0] == 1:
        st.error(f"**Nguy cơ mắc bệnh tiểu đường CAO!**")
        st.write(f"Xác suất mắc bệnh: **{prediction_proba[0][1]*100:.2f}%**")
        st.warning("Vui lòng tham khảo ý kiến bác sĩ để được tư vấn và kiểm tra chuyên sâu.")
    else:
        st.success(f"**Nguy cơ mắc bệnh tiểu đường THẤP.**")
        st.write(f"Xác suất không mắc bệnh: **{prediction_proba[0][0]*100:.2f}%**")
        st.info("Hãy tiếp tục duy trì lối sống lành mạnh!")

    st.write("---")
    st.write("### Các thông số đã nhập:")
    st.table(input_data)

# --- 5. Thông tin thêm (tùy chọn) ---
st.sidebar.header("Về ứng dụng")
st.sidebar.info("Ứng dụng này được phát triển như một bài tập về mô hình dự đoán AI. Kết quả chỉ mang tính chất tham khảo và không thay thế cho chẩn đoán y tế chuyên nghiệp.")
st.sidebar.text("Phiên bản: 1.0")



