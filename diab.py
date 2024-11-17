import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import joblib

def main():
    # Set page title and configuration
    st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About"])

    if page == "Home":
        home_page()
    elif page == "About":
        about_page()

def home_page():
    st.markdown("<div style='text-align:center'><h1 style='font-family:Ink Free;'>DiaBEAT!</h1></div>", unsafe_allow_html=True)
    st.write("<div style='text-align:center'><h4 style='font-family:Ink Free;'>Stay One Step Ahead of Diabetes.</h4></div>", unsafe_allow_html=True)
    # Centered header with Pricedown font
    image_path = "diab.webp"
    st.image(image_path, use_container_width=True)

    st.write("Enter the required information below to predict diabetes risk.")

    try:
        # Load your trained model
        classifier = joblib.load('diabetes_model.pkl')

        # Load the scaler
        scaler = joblib.load('scaler.pkl')

        # Create input form
        with st.form("prediction_form"):
            st.subheader("Patient Information")

            # Create two columns for better layout
            col1, col2 = st.columns(2)

            with col1:
                pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
                glucose = st.number_input("Glucose Level", min_value=0, max_value=500, value=100)
                blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
                skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

            with col2:
                insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=100)
                bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
                diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
                age = st.number_input("Age", min_value=0, max_value=120, value=30)

            submit_button = st.form_submit_button("Predict")

        # Make prediction when form is submitted
        if submit_button:
            # Create input array for prediction
            input_data = (pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, diabetes_pedigree, age)

            # Convert to numpy array
            input_data_as_numpy_array = np.asarray(input_data)

            # Reshape the array
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

            # Standardize the input data
            std_data = scaler.transform(input_data_reshaped)

            # Make prediction
            prediction = classifier.predict(std_data)

            # Display result with custom styling
            st.subheader("Prediction Result")
            if prediction[0] == 0:
                st.success("The person is not diabetic")
                st.markdown("""
                    <div style='background-color: #8EB89A;color:black; padding: 20px; border-radius: 5px;'>
                        ✅ Low Risk: The model predicts no diabetes based on the provided parameters.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("The person is diabetic")
                st.markdown("""
                    <div style='background-color: #DD5540;color:black; padding: 20px; border-radius: 5px;'>
                        ⚠️ High Risk: The model predicts diabetes based on the provided parameters.
                    </div>
                    """, unsafe_allow_html=True)

            # Display input values in a nice format
            st.subheader("Input Values")
            col1, col2 = st.columns(2)

            with col1:
                st.info("Patient Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness'],
                    'Value': [pregnancies, glucose, blood_pressure, skin_thickness]
                })
                st.table(metrics_df)

            with col2:
                st.info("Additional Information")
                info_df = pd.DataFrame({
                    'Metric': ['Insulin', 'BMI', 'Diabetes Pedigree', 'Age'],
                    'Value': [insulin, bmi, diabetes_pedigree, age]
                })
                st.table(info_df)

        # Add disclaimer
        st.markdown("""
        ---
        **Disclaimer**: This is a prediction based on machine learning and should not be used as a substitute
        for professional medical advice, diagnosis, or treatment. Please consult with a qualified healthcare
        provider for medical guidance.
        """)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please ensure the model and scaler are properly loaded.")

def about_page():
    st.title("About")
    st.write("DiaBEAT is a diabetes prediction system that uses a machine learning model to predict diabetes risk based on patient information using supervised Machine Learning. This project is done in Python.")
    st.write( "#### Credits:")
    image_path = "about.jpg"
    st.image(image_path)
    st.write( "Arnob Aich Anurag")
    st.write( "Research Intern at AMIR Lab (Advanced Machine Intelligence Research Lab)")
    st.write( "Associate Content Writer  at AIEC (AI Expert Career)")
    st.write( "Student at American International University Bangladesh")
    st.write( "Dhaka,Bangladesh")
    st.write( "For more information, please contact me at my email.")
    st.write("Email:aicharnob@gmail.com")

if __name__ == "__main__":
    main()
