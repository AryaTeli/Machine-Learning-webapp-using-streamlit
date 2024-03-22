import numpy as np
import pandas as pd
import streamlit as st 
import pickle

model = pickle.load(open(r'D:\ML_LAB\model_dev\models\gnb_company_bayes_model.pkl', 'rb'))

def main(): 
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">US-Market or Not Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    sales = st.text_input("Sales") 
    advertising = st.text_input("Advertising") 
    price = st.text_input("Price") 
    age = st.text_input("Age") 
    education = st.text_input("Education") 
    urban = st.text_input("Urban (1 for Yes, 0 for No)") 
    
    if st.button("Predict"): 
        data = {'Sales': float(sales), 'Advertising': float(advertising), 'Price': float(price), 'Age': int(age), 'Education': int(education), 'Urban': int(urban)}
        
        df = pd.DataFrame([data])
            
        prediction = model.predict(df)
    
        output = int(prediction[0])
        # if output == 1:
        #     text = "1"
        # else:
        #     text = "0"

        st.success(output)
      
if __name__=='__main__': 
    main()
