
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np


# ==========================================
# 1. KHỞI TẠO FLASK
# ==========================================

app = Flask(__name__)


# ==========================================
# 2. LOAD MODEL PIPELINE
# ==========================================

model = joblib.load("bank_marketing_pipeline.pkl")


# ==========================================
# 3. CẤU HÌNH MODEL
# ==========================================

# Threshold được chọn từ validation data
THRESHOLD = 0.20

# Phiên bản model
MODEL_VERSION = "bank-marketing-rf-v1.0"


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

def create_features(df):

    df = df.copy()

    # --------------------------------------
    # Feature 1: Nhóm tuổi
    # --------------------------------------

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, np.inf],
        labels=[
            "<=25",
            "26-35",
            "36-45",
            "46-55",
            "56-65",
            ">65"
        ]
    )

    # --------------------------------------
    # Feature 2: Đã từng được liên hệ
    # --------------------------------------

    df["previously_contacted"] = np.where(
        df["previous"] > 0,
        "yes",
        "no"
    )

    # --------------------------------------
    # Feature 3: Balance âm / dương
    # --------------------------------------

    df["balance_sign"] = np.where(
        df["balance"] >= 0,
        "positive",
        "negative"
    )

    # --------------------------------------
    # Feature 4: Campaign intensity
    # --------------------------------------

    df["campaign_intensity"] = pd.cut(
        df["campaign"],
        bins=[0, 1, 2, 4, np.inf],
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )

    # --------------------------------------
    # Feature 5: Nhóm số lần liên hệ trước
    # --------------------------------------

    df["previous_contact_group"] = pd.cut(
        df["previous"],
        bins=[-1, 0, 2, 5, np.inf],
        labels=[
            "None",
            "Few",
            "Several",
            "Many"
        ]
    )

    return df


# ==========================================
# 5. DANH SÁCH FEATURE GỐC
# ==========================================

required_features = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome"
]


# ==========================================
# 6. DANH SÁCH FEATURE SAU FEATURE ENGINEERING
# ==========================================

model_features = required_features + [
    "age_group",
    "previously_contacted",
    "balance_sign",
    "campaign_intensity",
    "previous_contact_group"
]


# ==========================================
# 7. API /predict
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------
        # Nhận JSON
        # ----------------------------------

        data = request.get_json()

        # Kiểm tra JSON
        if data is None:

            return jsonify({
                "error": "Request body phải là JSON"
            }), 400


        # ----------------------------------
        # Kiểm tra feature bắt buộc
        # ----------------------------------

        missing_features = [
            feature
            for feature in required_features
            if feature not in data
        ]


        if missing_features:

            return jsonify({
                "error": "Thiếu feature bắt buộc",
                "missing_features": missing_features
            }), 400


        # ----------------------------------
        # Chuyển JSON thành DataFrame
        # ----------------------------------

        input_df = pd.DataFrame([data])


        # ----------------------------------
        # Feature engineering
        # ----------------------------------

        input_df = create_features(input_df)


        # ----------------------------------
        # Giữ đúng thứ tự feature
        # ----------------------------------

        input_df = input_df[model_features]


        # ----------------------------------
        # Predict probability
        # ----------------------------------

        probability = float(
            model.predict_proba(input_df)[0, 1]
        )


        # ----------------------------------
        # Áp dụng threshold
        # ----------------------------------

        prediction = int(
            probability >= THRESHOLD
        )


        # ----------------------------------
        # Opportunity level
        # ----------------------------------

        if probability < THRESHOLD:

            opportunity_level = "Low"

        elif probability < 0.50:

            opportunity_level = "Medium"

        else:

            opportunity_level = "High"


        # ----------------------------------
        # Trả kết quả
        # ----------------------------------

        return jsonify({

            "prediction": prediction,

            "probability": round(
                probability,
                6
            ),

            "threshold": THRESHOLD,

            "model_version": MODEL_VERSION,

            "opportunity_level": opportunity_level

        })


    # ======================================
    # XỬ LÝ REQUEST LỖI
    # ======================================

    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 400


# ==========================================
# 8. CHẠY FLASK SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
