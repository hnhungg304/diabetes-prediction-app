# test_integration.py
import pytest
import pandas as pd
import numpy as np
import joblib
import os # Để kiểm tra sự tồn tại của file

# --- 1. Giả lập dữ liệu đầu vào (giả sử từ Google Sheet hoặc người dùng) ---
# Tên và thứ tự các cột PHẢI KHỚP chính xác với mô hình của bạn đã huấn luyện.
# Thứ tự đã được bạn cung cấp: Age, Sex, BMI, Waist_Circumference, Fasting_Blood_Glucose, HbA1c, Family_History_of_Diabetes, Previous_Gestational_Diabetes
EXPECTED_FEATURES = [
    'Age',
    'Sex',
    'BMI',
    'Waist_Circumference',
    'Fasting_Blood_Glucose',
    'HbA1c',
    'Family_History_of_Diabetes',
    'Previous_Gestational_Diabetes'
]

# Dữ liệu mẫu (chỉ là ví dụ, bạn có thể thay đổi để thử các trường hợp khác)
# 0: Không mắc bệnh, 1: Mắc bệnh
MOCK_PATIENT_DATA_HEALTHY = {
    'Age': 30,
    'Sex': 0, # Nam (hoặc mã hóa theo cách của bạn)
    'BMI': 22.0,
    'Waist_Circumference': 80,
    'Fasting_Blood_Glucose': 90,
    'HbA1c': 5.5,
    'Family_History_of_Diabetes': 0, # Không (hoặc mã hóa theo cách của bạn)
    'Previous_Gestational_Diabetes': 0 # Không (hoặc mã hóa theo cách của bạn)
}

MOCK_PATIENT_DATA_AT_RISK = {
    'Age': 55,
    'Sex': 1, # Nữ (hoặc mã hóa theo cách của bạn)
    'BMI': 35.0,
    'Waist_Circumference': 105,
    'Fasting_Blood_Glucose': 140,
    'HbA1c': 7.2,
    'Family_History_of_Diabetes': 1, # Có (hoặc mã hóa theo cách của bạn)
    'Previous_Gestational_Diabetes': 1 # Có (hoặc mã hóa theo cách của bạn)
}

# --- 2. Tải mô hình một lần cho tất cả các bài kiểm thử ---
# Fixture này sẽ chạy một lần và cung cấp mô hình cho các bài kiểm thử khác
@pytest.fixture(scope="module")
def loaded_model():
    model_path = 'random_forest_diabetes_model.joblib'
    if not os.path.exists(model_path):
        pytest.fail(f"Lỗi: Không tìm thấy tệp mô hình tại '{model_path}'. Hãy đảm bảo bạn đã tải lên file mô hình vào Colab.")
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        pytest.fail(f"Không thể tải mô hình từ '{model_path}'. Lỗi: {e}")

# --- 3. Các Test Case Tích hợp ---

def test_model_loading_and_prediction_healthy_case(loaded_model):
    """
    Kiểm thử tích hợp: Tải mô hình và dự đoán trường hợp bệnh nhân khỏe mạnh.
    Kiểm tra xem mô hình có trả về dự đoán hợp lệ (0 hoặc 1) và xác suất hợp lý.
    """
    print("\n--- Running test_model_loading_and_prediction_healthy_case ---")
    
    # Chuyển đổi dữ liệu mẫu sang DataFrame với đúng thứ tự cột
    input_df = pd.DataFrame([MOCK_PATIENT_DATA_HEALTHY], columns=EXPECTED_FEATURES)
    print(f"Input DataFrame for Healthy Case:\n{input_df}")

    # Thực hiện dự đoán
    prediction = loaded_model.predict(input_df)
    prediction_proba = loaded_model.predict_proba(input_df)

    print(f"Prediction for Healthy Case: {prediction[0]}")
    print(f"Prediction Probability for Healthy Case: {prediction_proba[0]}")

    # Kiểm tra kết quả dự đoán
    assert prediction.shape == (1,) # Đảm bảo trả về 1 dự đoán
    assert prediction[0] in [0, 1] # Đảm bảo dự đoán là 0 hoặc 1
    # Đối với trường hợp khỏe mạnh, mong đợi xác suất mắc bệnh thấp
    assert prediction_proba[0][1] < 0.5 # Xác suất lớp 1 (mắc bệnh) phải nhỏ hơn 0.5


def test_model_loading_and_prediction_at_risk_case(loaded_model):
    """
    Kiểm thử tích hợp: Tải mô hình và dự đoán trường hợp bệnh nhân có nguy cơ cao.
    Kiểm tra xem mô hình có trả về dự đoán hợp lệ và xác suất hợp lý.
    """
    print("\n--- Running test_model_loading_and_prediction_at_risk_case ---")

    # Chuyển đổi dữ liệu mẫu sang DataFrame với đúng thứ tự cột
    input_df = pd.DataFrame([MOCK_PATIENT_DATA_AT_RISK], columns=EXPECTED_FEATURES)
    print(f"Input DataFrame for At-Risk Case:\n{input_df}")

    # Thực hiện dự đoán
    prediction = loaded_model.predict(input_df)
    prediction_proba = loaded_model.predict_proba(input_df)

    print(f"Prediction for At-Risk Case: {prediction[0]}")
    print(f"Prediction Probability for At-Risk Case: {prediction_proba[0]}")

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]
    # Đối với trường hợp có nguy cơ, mong đợi xác suất mắc bệnh cao
    assert prediction_proba[0][1] > 0.5 # Xác suất lớp 1 (mắc bệnh) phải lớn hơn 0.5


def test_input_dataframe_structure():
    """
    Kiểm thử tích hợp: Đảm bảo cấu trúc DataFrame đầu vào khớp với mong đợi.
    Kiểm tra tên cột và thứ tự.
    """
    print("\n--- Running test_input_dataframe_structure ---")
    test_data = {
        'Age': 40,
        'Sex': 0,
        'BMI': 28.0,
        'Waist_Circumference': 95,
        'Fasting_Blood_Glucose': 110,
        'HbA1c': 6.0,
        'Family_History_of_Diabetes': 0,
        'Previous_Gestational_Diabetes': 0
    }
    
    # Tạo DataFrame từ dữ liệu test với thứ tự cột đã xác định
    input_df = pd.DataFrame([test_data], columns=EXPECTED_FEATURES)
    
    print(f"Created Input DataFrame Columns: {input_df.columns.tolist()}")
    print(f"Expected Features: {EXPECTED_FEATURES}")

    # Kiểm tra xem tên cột và thứ tự có khớp không
    assert input_df.columns.tolist() == EXPECTED_FEATURES, "Tên cột hoặc thứ tự không khớp với mô hình"
    
    # Kiểm tra kiểu dữ liệu (tùy chọn, nếu bạn có yêu cầu nghiêm ngặt về kiểu)
    # Ví dụ: assert input_df['Age'].dtype == 'int64'

# --- Hướng dẫn kiểm thử kết nối Google Sheet (Conceptual) ---
# Để kiểm thử kết nối THỰC TẾ với Google Sheet trên Colab, bạn sẽ cần:
# 1. Cài đặt thư viện: !pip install gspread google-auth
# 2. Cấu hình xác thực: Thường là Service Account Key file (tệp JSON)
#    - Bạn sẽ cần tạo một Service Account trên Google Cloud Console,
#      cấp quyền truy cập vào Google Sheet, và tải tệp JSON key về.
#    - Tải tệp JSON key đó lên Colab.
# 3. Code đọc dữ liệu:
# import gspread
# from google.oauth2.service_account import Credentials

# def get_data_from_google_sheet(sheet_name, worksheet_name, key_file_path):
#     try:
#         # Sử dụng Scope cần thiết cho Google Sheets API
#         scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
#         creds = Credentials.from_service_account_file(key_file_path, scopes=scope)
#         client = gspread.authorize(creds)
#         sheet = client.open(sheet_name).worksheet(worksheet_name)
#         data = sheet.get_all_records() # Lấy tất cả dữ liệu dưới dạng list of dicts
#         return pd.DataFrame(data)
#     except Exception as e:
#         print(f"Lỗi khi đọc Google Sheet: {e}")
#         return None

# # Ví dụ về test case cho Google Sheet (sẽ không chạy tự động nếu không có cấu hình thật)
# def test_google_sheet_connection_colab():
#     """
#     Kiểm thử kết nối với Google Sheet (yêu cầu cấu hình xác thực thực tế và tải key file lên Colab).
#     """
#     print("\n--- Running test_google_sheet_connection_colab ---")
#     # Thay thế bằng tên sheet, worksheet và đường dẫn key file của bạn (trong môi trường Colab)
#     SHEET_NAME = "MyDiabetesData"
#     WORKSHEET_NAME = "Patients"
#     KEY_FILE = "your_service_account_key.json" # Tệp này phải được tải lên Colab

#     df_from_sheet = get_data_from_google_sheet(SHEET_NAME, WORKSHEET_name, KEY_FILE)
#     assert df_from_sheet is not None, "Không thể kết nối hoặc đọc dữ liệu từ Google Sheet"
#     assert not df_from_sheet.empty, "Google Sheet trống rỗng"
#     # Thêm các kiểm tra khác về cấu trúc cột, kiểu dữ liệu, v.v.
#     # assert all(col in df_from_sheet.columns for col in EXPECTED_FEATURES)
