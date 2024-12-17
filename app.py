import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load Dataset
@st.cache_resource
def load_and_train_model():
    file_path = 'synthetic_final_mapping (1).csv'
    data = pd.read_csv(file_path)

    # Select relevant columns for the model
    relevant_columns = [
        "Role Status", "Region", "Project Type", "Track", "Location Shore", 
        "Primary Skill (Must have)", "Grade", "Employment ID"
    ]
    data = data[relevant_columns]

    # Preprocess data
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].fillna("Unknown"))
        label_encoders[column] = le

    # Train the model
    X = data.drop("Employment ID", axis=1)
    y = data["Employment ID"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, data, label_encoders

# Recommend Employees
def recommend_employees(model, input_data, data):
    predictions = model.predict_proba([input_data])[0]
    employee_indices = predictions.argsort()[-3:][::-1]
    employee_ids = data["Employment ID"].unique()
    top_employees = [employee_ids[i] for i in employee_indices]
    return top_employees

# Streamlit App Configuration
st.set_page_config(page_title="Demand to Employee", layout="wide")

# Add custom CSS for background, text and inputs styling
st.markdown(
    """
    <style>
    /* Set black background */
    body {
        background-color: #000000;
        color: white;
        font-family: 'Arial', sans-serif;
    }

    /* Style for the main title */
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding-top: 50px;
        font-family: 'Helvetica', sans-serif;
    }

    /* Style for subheaders */
    .header {
        font-size: 22px;
        color: #ffffff;
        font-family: 'Verdana', sans-serif;
        text-align: center;
        background-color: rgba(0, 0, 0, 0.7);  /* Semi-transparent black background for better readability */
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }

    /* Style for the input section */
    .background-pattern {
        background-color: #333333;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        margin-top: 30px;
    }

    /* Input fields and selectbox styling */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #222222;
        border: 2px solid #444444;
        color: white;
        padding: 10px;
        border-radius: 8px;
        font-size: 16px;
    }

    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #00bcd4;
    }

    /* Buttons Styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    /* Footer Styling */
    footer {
        font-size: 14px;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True
)

# Title with a larger font and custom styling
st.markdown('<p class="title">🚀 Demand to Employee</p>', unsafe_allow_html=True)

# Header text with background color to ensure visibility
st.markdown('<p class="header">An AI-based system for HR to match new project demands with the best employees.</p>', unsafe_allow_html=True)

# Load and train model
model, data, label_encoders = load_and_train_model()

# Collect Demand Attributes
st.markdown('<div class="background-pattern">', unsafe_allow_html=True)

# Create two columns for input layout
st.subheader("📊 Enter Project Demand Attributes")
user_input = []
columns = data.columns.drop("Employment ID")

# Create two columns for input layout
col1, col2 = st.columns(2)

# Distribute the input fields between the two columns
for idx, column in enumerate(columns):
    with col1 if idx % 2 == 0 else col2:
        if column in label_encoders:
            options = label_encoders[column].classes_
            value = st.selectbox(f"{column}:", options, key=column)
            user_input.append(label_encoders[column].transform([value])[0])
        else:
            value = st.number_input(f"{column}:", min_value=0, step=1, key=column)
            user_input.append(value)

# Add a button with a color to trigger the recommendation
if st.button("Get Suitable Employees"):
    try:
        recommendations = recommend_employees(model, user_input, data)
        st.subheader("🌟 Top 3 Recommended Employees")
        for i, employee in enumerate(recommendations, 1):
            st.write(f"**{i}. Employee ID:** {employee}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Footer with additional information or links
st.markdown("---")
st.markdown(
    """
    <footer>
        <p style="color:#8A2BE2;">Developed for HR Department to optimize project staffing.</p>
        <p>For any inquiries, contact HR Support. &copy; 2024</p>
    </footer>
    """, unsafe_allow_html=True
)
