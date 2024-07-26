import numpy as np
import pandas as pd
import streamlit as st
import pickle
import requests

# Replace with the raw URL of the file
url = 'https://raw.githubusercontent.com/AryaTeli/Machine-Learning-webapp-using-streamlit/main/model_code/gnb_company_bayes_model.pkl'

# Download the file
response = requests.get(url)

# Load the model directly
model = pickle.load(response.content)

def main():
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">US-Market or Not Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    sales = st.number_input("Sales", step=1.0)
    advertising = st.number_input("Advertising", step=1.0)
    price = st.number_input("Price", step=1.0)
    age = st.number_input("Age", step=1, format="%d")
    education = st.number_input("Education", step=1, format="%d")
    urban = st.selectbox("Urban", ["Yes", "No"])

    if st.button("Predict"):
        data = {'Sales': sales, 'Advertising': advertising, 'Price': price, 'Age': int(age), 'Education': int(education), 'Urban': 1 if urban == "Yes" else 0}
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        output = int(prediction[0])
        st.success(f"The prediction is: {'US Market' if output == 1 else 'Not US Market'}")

if __name__ == '__main__':
    main()
