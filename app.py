import os
import csv
from datetime import datetime

import torch
import torch.nn as nn

import streamlit as st

from PIL import Image

from torchvision import models, transforms


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Pneumonia Detection | MOULYA PERUMAL .U",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

MODEL_PATH = "model/pneumonia_resnet50.pth"

UPLOAD_FOLDER = "uploads"

CSV_FILE = "predictions.csv"

IMAGE_SIZE = 224


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    @keyframes floatCard {

        0% {
            transform: translateY(0px);
        }

        50% {
            transform: translateY(-8px);
        }

        100% {
            transform: translateY(0px);
        }

    }


    @keyframes gradientMove {

        0% {
            background-position: 0% 50%;
        }

        50% {
            background-position: 100% 50%;
        }

        100% {
            background-position: 0% 50%;
        }

    }


    @keyframes fadeIn {

        from {
            opacity: 0;
            transform: translateY(20px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }

    }


    .stApp {

        background:

        radial-gradient(
            circle at 10% 20%,
            rgba(99, 102, 241, 0.25),
            transparent 30%
        ),

        radial-gradient(
            circle at 90% 80%,
            rgba(6, 182, 212, 0.22),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #050816,
            #0f172a,
            #111827
        );

        background-size: 200% 200%;

        animation:
            gradientMove 15s ease infinite;

    }


    .main-title {

        font-size: 42px;

        font-weight: 800;

        text-align: center;

        color: white;

        animation:
            fadeIn 1s ease;

    }


    .subtitle {

        text-align: center;

        color: #cbd5e1;

        font-size: 17px;

        margin-bottom: 25px;

    }


    .brand {

        text-align: center;

        color: #a78bfa;

        font-size: 16px;

        font-weight: 700;

        letter-spacing: 2px;

    }


    .glass-card {

        background:
            rgba(255,255,255,0.08);

        border:
            1px solid
            rgba(255,255,255,0.16);

        border-radius: 24px;

        padding: 25px;

        backdrop-filter: blur(20px);

        box-shadow:
            0 20px 60px
            rgba(0,0,0,0.35);

        animation:
            fadeIn 0.8s ease;

    }


    .result-card {

        background:
            rgba(255,255,255,0.08);

        border:
            1px solid
            rgba(255,255,255,0.16);

        border-radius: 24px;

        padding: 30px;

        text-align: center;

        animation:
            floatCard 4s ease-in-out infinite;

    }


    .prediction-normal {

        color: #22c55e;

        font-size: 38px;

        font-weight: 800;

    }


    .prediction-pneumonia {

        color: #f87171;

        font-size: 38px;

        font-weight: 800;

    }


    .confidence {

        font-size: 25px;

        color: #e2e8f0;

        font-weight: 700;

    }


    .footer {

        text-align: center;

        color: #94a3b8;

        margin-top: 40px;

        padding: 20px;

    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        return None


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )


    model = models.resnet50(
        weights=None
    )


    number_of_features = (
        model.fc.in_features
    )


    model.fc = nn.Sequential(

        nn.Linear(
            number_of_features,
            256
        ),

        nn.ReLU(),

        nn.Dropout(
            0.4
        ),

        nn.Linear(
            256,
            2
        )

    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model = model.to(device)

    model.eval()


    classes = checkpoint.get(
        "classes",
        ["NORMAL", "PNEUMONIA"]
    )


    return model, classes


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

image_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])


# =========================================================
# SAVE PREDICTION TO CSV
# =========================================================

def save_prediction(
    filename,
    prediction,
    confidence
):

    file_exists = os.path.exists(
        CSV_FILE
    )


    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )


        if not file_exists:

            writer.writerow([

                "timestamp",
                "filename",
                "prediction",
                "confidence",
                "device"

            ])


        writer.writerow([

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            filename,

            prediction,

            round(
                confidence,
                2
            ),

            str(device)

        ])


# =========================================================
# HEADER
# =========================================================

st.markdown(

    """
    <div class="brand">
    MOULYA PERUMAL .U
    </div>

    <div class="main-title">
    🩺 AUTOMATED PNEUMONIA DETECTION
    </div>

    <div class="subtitle">
    Chest X-Ray Classification using
    ResNet-50 Transfer Learning
    </div>
    """,

    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

model_data = load_model()


if model_data is None:

    st.error(
        """
        Model file not found.

        Please train the model first using:

        python train_model.py
        """
    )

    st.stop()


model, class_names = model_data


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## 🩺 Project Information"
    )

    st.write(
        "**Model:** ResNet-50"
    )

    st.write(
        "**Technique:** Transfer Learning"
    )

    st.write(
        "**Classes:** Normal / Pneumonia"
    )

    st.write(
        "**Input:** Chest X-Ray"
    )

    st.write(
        "**Image Size:** 224 × 224"
    )

    st.write(
        f"**Device:** {device}"
    )

    st.divider()

    st.markdown(
        "### 📊 Power BI"
    )

    st.write(
        """
        Every prediction is automatically
        stored in predictions.csv.
        """
    )


# =========================================================
# MAIN COLUMNS
# =========================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# =========================================================
# LEFT SIDE
# =========================================================

with left_column:

    st.markdown(

        """
        <div class="glass-card">

        <h2>📤 Upload Chest X-Ray</h2>

        <p>
        Upload a chest X-ray image
        for classification.
        </p>

        </div>
        """,

        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(

        "Choose an X-Ray image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ],

        label_visibility="collapsed"

    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert(
            "RGB"
        )


        st.image(
            image,
            caption="Uploaded Chest X-Ray",
            use_container_width=True
        )


        analyze_button = st.button(

            "🔍 ANALYZE X-RAY",

            type="primary",

            use_container_width=True

        )

    else:

        analyze_button = False


# =========================================================
# RIGHT SIDE
# =========================================================

with right_column:

    st.markdown(

        """
        <div class="glass-card">

        <h2>🤖 AI Prediction</h2>

        <p>
        ResNet-50 analyzes the uploaded
        X-ray image.
        </p>

        </div>
        """,

        unsafe_allow_html=True
    )


    if analyze_button:

        with st.spinner(
            "Analyzing X-Ray..."
        ):

            input_tensor = (

                image_transform(image)
                .unsqueeze(0)
                .to(device)

            )


            with torch.no_grad():

                outputs = model(
                    input_tensor
                )


                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )[0]


                prediction_index = int(
                    torch.argmax(
                        probabilities
                    )
                )


                confidence = float(

                    probabilities[
                        prediction_index
                    ] * 100

                )


                prediction = (

                    class_names[
                        prediction_index
                    ]

                )


        # =================================================
        # SAVE UPLOADED IMAGE
        # =================================================

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )


        saved_filename = (

            f"{timestamp}_"
            f"{uploaded_file.name}"

        )


        saved_path = os.path.join(

            UPLOAD_FOLDER,

            saved_filename

        )


        image.save(
            saved_path
        )


        # =================================================
        # SAVE CSV
        # =================================================

        save_prediction(

            uploaded_file.name,

            prediction,

            confidence

        )


        # =================================================
        # DISPLAY RESULT
        # =================================================

        if prediction.upper() == "NORMAL":

            result_class = (
                "prediction-normal"
            )

            icon = "✅"

        else:

            result_class = (
                "prediction-pneumonia"
            )

            icon = "⚠️"


        st.markdown(

            f"""
            <div class="result-card">

                <div style="font-size:55px;">
                {icon}
                </div>

                <div class="{result_class}">
                {prediction.upper()}
                </div>

                <br>

                <div class="confidence">
                Confidence: {confidence:.2f}%
                </div>

            </div>
            """,

            unsafe_allow_html=True
        )


        st.write("")


        st.subheader(
            "📊 Class Probabilities"
        )


        for index, class_name in enumerate(
            class_names
        ):

            probability = float(

                probabilities[index] * 100

            )


            st.progress(

                probability / 100,

                text=(
                    f"{class_name}: "
                    f"{probability:.2f}%"
                )

            )


# =========================================================
# ANALYTICS
# =========================================================

st.divider()

st.markdown(
    "## 📊 Prediction Analytics"
)


if os.path.exists(CSV_FILE):

    import pandas as pd


    prediction_data = pd.read_csv(
        CSV_FILE
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Total Predictions",
            len(prediction_data)
        )


    with col2:

        normal_count = (

            prediction_data[
                "prediction"
            ]
            .str.upper()
            .eq("NORMAL")
            .sum()

        )


        st.metric(
            "Normal",
            normal_count
        )


    with col3:

        pneumonia_count = (

            prediction_data[
                "prediction"
            ]
            .str.upper()
            .eq("PNEUMONIA")
            .sum()

        )


        st.metric(
            "Pneumonia",
            pneumonia_count
        )


    st.dataframe(

        prediction_data,

        use_container_width=True

    )


    csv_data = prediction_data.to_csv(
        index=False
    )


    st.download_button(

        label="⬇️ Download Predictions CSV",

        data=csv_data,

        file_name="predictions.csv",

        mime="text/csv"

    )

else:

    st.info(
        "Prediction history will appear "
        "after you analyze X-Ray images."
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.warning(

    """
    ⚠️ Educational Project Disclaimer:

    This application is a college deep-learning
    project. It is not a medical diagnostic system.
    The prediction should not be used for medical
    decisions. A qualified healthcare professional
    must evaluate medical images.
    """

)


# =========================================================
# FOOTER
# =========================================================

st.markdown(

    """
    <div class="footer">

    Automated Pneumonia Detection<br>

    Transfer Learning with ResNet-50<br>

    Designed by <b>MOULYA PERUMAL .U</b>

    </div>
    """,

    unsafe_allow_html=True
)