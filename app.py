
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessor
model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("churn_preprocessor.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    color: #17365D;
}
div.stButton > button {
    background-color: #17365D;
    color: white;
    border-radius: 8px;
    height: 3rem;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📊 Churn Predictor")
    st.write("Customer Analytics System")
    st.divider()
    st.info(
        "This system uses Machine Learning "
        "to predict whether a telecommunications "
        "customer is likely to leave."
    )
    st.caption("Model: Logistic Regression")
    st.caption("Dataset: IBM Telco Customer Churn")

# Main heading
st.title("Customer Churn Prediction System")
st.write(
    "Enter customer information below to "
    "estimate the likelihood of customer churn."
)

st.divider()

# Input form
with st.form("customer_form"):

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox(
            "Has Partner?", ["Yes", "No"]
        )

    with col2:
        dependents = st.selectbox(
            "Has Dependents?", ["Yes", "No"]
        )
        tenure = st.slider(
            "Tenure (months)", 0, 72, 12
        )
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

    with col3:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

    st.divider()
    st.subheader("🌐 Subscribed Services")

    col4, col5, col6 = st.columns(3)

    with col4:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

    with col5:
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with col6:
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    st.divider()
    st.subheader("💳 Contract and Billing")

    col7, col8, col9 = st.columns(3)

    with col7:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col8:
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

    with col9:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col10, col11 = st.columns(2)

    with col10:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

    with col11:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0
        )

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )

# Prediction
if submitted:

    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    processed_data = preprocessor.transform(customer_data)

    prediction = model.predict(processed_data)[0]

    probabilities = model.predict_proba(processed_data)[0]
    churn_index = list(model.classes_).index("Yes")
    churn_probability = probabilities[churn_index]

    st.divider()
    st.header("📈 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Churn Probability",
            f"{churn_probability * 100:.1f}%"
        )

        st.progress(float(churn_probability))

    with col2:
        st.metric(
            "Predicted Outcome",
            "Likely to Churn" if prediction == "Yes"
            else "Likely to Stay"
        )

    if prediction == "Yes":
        st.error(
            "⚠️ The customer is likely to churn. "
            "Consider appropriate customer retention measures."
        )
    else:
        st.success(
            "✅ The customer is likely to stay. "
            "Continue providing good customer service."
        )

    st.caption(
        "Note: Predictions are estimates produced by "
        "the trained machine learning model and are "
        "not guarantees of customer behaviour."
    )
