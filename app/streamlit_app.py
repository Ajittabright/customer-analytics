# =============================================================================
# app/streamlit_app.py
# =============================================================================
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

from src.data_loader import load_config
from src.predictor import predict_single_customer, load_model_artifacts

st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_config():
    return load_config()

@st.cache_resource
def get_model_artifacts():
    config = get_config()
    return load_model_artifacts(config)

def apply_custom_styles():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #3498db;
        margin-bottom: 2rem;
    }
    .result-box-churn {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .result-box-stay {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9rem;
        padding: 2rem 0;
        border-top: 1px solid #ecf0f1;
        margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    st.sidebar.markdown("### Customer Information")
    st.sidebar.markdown("---")

    st.sidebar.markdown("#### Personal Details")
    age = st.sidebar.slider("Age", min_value=18, max_value=92, value=35)
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
    geography = st.sidebar.selectbox("Geography", options=["France", "Germany", "Spain"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Financial Details")
    credit_score = st.sidebar.slider("Credit Score", min_value=300, max_value=850, value=650)
    balance = st.sidebar.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
    estimated_salary = st.sidebar.number_input("Estimated Salary ($)", min_value=0.0, max_value=300000.0, value=80000.0, step=1000.0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Banking Details")
    tenure = st.sidebar.slider("Tenure (Years)", min_value=0, max_value=10, value=5)
    num_products = st.sidebar.selectbox("Number of Products", options=[1, 2, 3, 4], index=1)
    has_credit_card = st.sidebar.checkbox("Has Credit Card", value=True)
    is_active = st.sidebar.checkbox("Is Active Member", value=True)

    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("PREDICT CHURN", use_container_width=True, type="primary")

    customer_data = {
        "CreditScore":     credit_score,
        "Geography":       geography,
        "Gender":          gender,
        "Age":             age,
        "Tenure":          tenure,
        "Balance":         balance,
        "NumOfProducts":   num_products,
        "HasCrCard":       int(has_credit_card),
        "IsActiveMember":  int(is_active),
        "EstimatedSalary": estimated_salary
    }

    return customer_data, predict_button

def render_header():
    st.markdown("<div class='main-header'>Customer Analytics and Churn Prediction</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    try:
        artifacts = get_model_artifacts()
        model_name = artifacts["metadata"]["model_name"]
        feature_count = artifacts["metadata"]["feature_count"]
        with col1:
            st.metric(label="Active Model", value=model_name)
        with col2:
            st.metric(label="Features Used", value=str(feature_count))
        with col3:
            st.metric(label="Last Updated", value=datetime.now().strftime("%H:%M:%S"))
    except Exception:
        st.warning("Model not loaded. Please run model_trainer.py first!")

def render_prediction_results(result, customer_data):
    st.markdown("---")
    st.markdown("## Prediction Results")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if result["will_churn"]:
            st.markdown(
                "<div class='result-box-churn'><h1>CHURN PREDICTED</h1><h2>This customer is likely to leave!</h2></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box-stay'><h1>CUSTOMER WILL STAY</h1><h2>This customer is likely to remain!</h2></div>",
                unsafe_allow_html=True
            )

    with col2:
        st.metric(
            label="Churn Probability",
            value=str(result["churn_probability"]) + "%",
            delta=str(round(result["churn_probability"] - 20, 1)) + "% vs avg",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            label="Stay Probability",
            value=str(result["stay_probability"]) + "%"
        )

    st.markdown("### Risk Level: " + result["risk_emoji"] + " " + result["risk_level"])

    st.markdown("### Probability Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 3))
        categories = ["Will Stay", "Will Churn"]
        values = [result["stay_probability"], result["churn_probability"]]
        colors = ["#2ecc71", "#e74c3c"]
        bars = ax.barh(categories, values, color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                str(round(val, 1)) + "%",
                va="center",
                fontweight="bold",
                fontsize=12
            )
        ax.set_xlim(0, 115)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Churn vs Stay Probability", fontweight="bold")
        ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### Customer Summary")
        summary_data = {
            "Attribute": ["Age", "Geography", "Gender", "Credit Score", "Balance", "Salary", "Tenure", "Products", "Active Member"],
            "Value": [
                str(customer_data["Age"]) + " years",
                customer_data["Geography"],
                customer_data["Gender"],
                str(customer_data["CreditScore"]),
                "$" + str(round(customer_data["Balance"], 2)),
                "$" + str(round(customer_data["EstimatedSalary"], 2)),
                str(customer_data["Tenure"]) + " years",
                str(customer_data["NumOfProducts"]),
                "Yes" if customer_data["IsActiveMember"] else "No"
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Business Recommendation")
    if result["will_churn"]:
        st.error("ALERT: " + result["recommendation"])
    else:
        st.success("INFO: " + result["recommendation"])

    st.markdown("---")
    st.markdown("### Export Prediction")

    export_data = {
        "CreditScore": customer_data["CreditScore"],
        "Geography": customer_data["Geography"],
        "Gender": customer_data["Gender"],
        "Age": customer_data["Age"],
        "Tenure": customer_data["Tenure"],
        "Balance": customer_data["Balance"],
        "NumOfProducts": customer_data["NumOfProducts"],
        "HasCrCard": customer_data["HasCrCard"],
        "IsActiveMember": customer_data["IsActiveMember"],
        "EstimatedSalary": customer_data["EstimatedSalary"],
        "Predicted_Churn": result["prediction"],
        "Churn_Probability_%": result["churn_probability"],
        "Stay_Probability_%": result["stay_probability"],
        "Risk_Level": result["risk_level"],
        "Recommendation": result["recommendation"],
        "Model_Used": result["model_name"],
        "Prediction_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    export_df = pd.DataFrame([export_data])
    csv = export_df.to_csv(index=False)

    st.download_button(
        label="Download Prediction as CSV",
        data=csv,
        file_name="churn_prediction_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv",
        mime="text/csv",
        use_container_width=True
    )

def render_welcome():
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## Welcome to Customer Analytics!")
        st.markdown("This dashboard helps banks predict which customers are likely to leave so they can take action to retain them.")
        st.markdown("### How to use:")
        st.markdown("1. Fill in customer details in the sidebar")
        st.markdown("2. Click PREDICT CHURN button")
        st.markdown("3. View prediction results and recommendations")
        st.markdown("4. Export results as CSV if needed")

    with col2:
        st.markdown("## Model Performance")
        perf_data = {
            "Metric": ["Algorithm", "Training Data", "Test Data", "Features Used"],
            "Value": ["Gradient Boosting", "8,000 customers", "2,000 customers", "20+"]
        }
        st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)

        st.markdown("## Risk Levels")
        risk_data = {
            "Level": ["Low Risk", "Medium Risk", "High Risk"],
            "Churn Probability": ["Less than 30%", "30% to 60%", "More than 60%"]
        }
        st.dataframe(pd.DataFrame(risk_data), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("## Key Business Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("Germany has the highest churn rate around 32 percent. Focus retention efforts there!")
    with col2:
        st.warning("Customers aged 45 to 60 are most likely to churn. Target this age group!")
    with col3:
        st.error("Customers with 3 to 4 products have surprisingly high churn rates!")

def main():
    apply_custom_styles()
    render_header()
    customer_data, predict_button = render_sidebar()

    if predict_button:
        with st.spinner("Analyzing customer data..."):
            try:
                config = get_config()
                result = predict_single_customer(customer_data, config)
                render_prediction_results(result, customer_data)
            except FileNotFoundError:
                st.error("Model files not found! Please run model_trainer.py first.")
            except Exception as e:
                st.error("Prediction error: " + str(e))
    else:
        render_welcome()

    st.markdown("<div class='footer'>Built with Python, Scikit-learn and Streamlit | Customer Analytics v1.0</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
