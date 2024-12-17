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

# Streamlit App
st.set_page_config(page_title="Demand to Talent", layout="wide")

# Add a header with an icon and description
st.markdown(
    """
    <style>
    .title {
        font-size: 30px;
        font-weight: bold;
        color: #1E90FF;
    }
    .header {
        font-size: 20px;
        color: #2E8B57;
    }
    .subheader {
        font-size: 18px;
        color: #FF6347;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<p class="title">🚀 Demand To Talent</p>', unsafe_allow_html=True)
st.markdown('<p class="header">An AI-based system for HR to match new project demands with the best employees.</p>', unsafe_allow_html=True)

# Load and train model
model, data, label_encoders = load_and_train_model()

# Collect Demand Attributes
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
button_style = """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

if st.button("Get Suitable Employees"):
    try:
        recommendations = recommend_employees(model, user_input, data)
        st.subheader("🌟 Top 3 Employees")
        for i, employee in enumerate(recommendations, 1):
            st.write(f"**{i}. Employee ID:** {employee}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer with additional information or links
st.markdown("---")
st.markdown(
    """
    <footer style="text-align:center;">
        <p style="color:#8A2BE2; font-size:14px;">Developed for HR Department to optimize project staffing.</p>
        <p style="font-size:12px;">For any inquiries, contact HR Support. &copy; 2024</p>
    </footer>
    """, unsafe_allow_html=True
)
