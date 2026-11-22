# ============================================================
# DIAMOND PRICE PREDICTION USING KNN REGRESSION
# STREAMLIT WEB APPLICATION
# ============================================================

# 1. Import Required Libraries

import streamlit as st
import pandas as pd
import pickle

from pathlib import Path


# 2. Define Project Directory

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "knn_diamond_pipeline.pkl"


# 3. Streamlit Page Configuration

st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)


# 4. Load Trained Machine Learning Model

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as file:

        model = pickle.load(file)

    return model


# 5. Application Title

st.title("💎 Diamond Price Prediction")

st.subheader("Predict Diamond Prices Using Machine Learning")

st.write(
    "Enter the characteristics of your diamond "
    "to predict its estimated market price."
)

st.divider()


# 6. Create User Input Form

st.header("Diamond Characteristics")

with st.form("diamond_prediction_form"):

    # Carat

    carat = st.number_input(
        "Carat Weight",
        min_value=0.20,
        max_value=5.01,
        value=1.00,
        step=0.01
    )

    # Cut

    cut = st.selectbox(
        "Diamond Cut",
        ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    )

    # Color

    color = st.selectbox(
        "Diamond Color",
        ["D", "E", "F", "G", "H", "I", "J"]
    )

    # Clarity

    clarity = st.selectbox(
        "Diamond Clarity",
        ["I1", "SI2", "SI1", "VS2", "VS1",
         "VVS2", "VVS1", "IF"]
    )

    # Depth

    depth = st.number_input(
        "Depth Percentage",
        min_value=40.0,
        max_value=80.0,
        value=61.5,
        step=0.1
    )

    # Table

    table = st.number_input(
        "Table Percentage",
        min_value=40.0,
        max_value=80.0,
        value=57.0,
        step=0.1
    )

    # Dimensions

    st.subheader("Diamond Dimensions (mm)")

    x = st.number_input(
        "Length (x)",
        min_value=0.0,
        max_value=11.0,
        value=5.0,
        step=0.01
    )

    y = st.number_input(
        "Width (y)",
        min_value=0.0,
        max_value=60.0,
        value=5.0,
        step=0.01
    )

    z = st.number_input(
        "Depth (z)",
        min_value=0.0,
        max_value=32.0,
        value=3.0,
        step=0.01
    )

    # Prediction Button

    submit = st.form_submit_button(
        "Predict Diamond Price 💎",
        use_container_width=True
    )


# 7. Prediction Logic

if submit:

    if x <= 0 or y <= 0 or z <= 0:

        st.error("Diamond dimensions must be greater than zero.")

    else:

        try:

            model = load_model()

            # Create Input DataFrame

            input_data = pd.DataFrame({

                "carat": [carat],

                "cut": [cut],

                "color": [color],

                "clarity": [clarity],

                "depth": [depth],

                "table": [table],

                "x": [x],

                "y": [y],

                "z": [z]

            })


            # Predict Price

            prediction = model.predict(input_data)[0]


            # Display Result

            st.success("Prediction Completed Successfully!")

            st.divider()

            st.subheader("Estimated Diamond Price")

            st.metric(
                label="Predicted Price",
                value=f"${prediction:,.2f}"
            )

            st.info(
                "This is a Machine Learning-based estimate, "
                "not a guaranteed market selling price."
            )


        except FileNotFoundError:

            st.error(
                "Trained model not found. "
                "Please execute src/train.py first."
            )

        except Exception as e:

            st.error(f"Prediction Error: {e}")


# 8. Application Footer

st.divider()

st.caption(
    "Diamond Price Prediction | "
    "Machine Learning Project using KNN Regression"
)