import streamlit as st
st.set_page_config(page_title="Burnout Risk App", layout="wide")  # ✅ Must come right after importing Streamlit

import edamd as eda
import prediction 

# Sidebar navigation
with st.sidebar:
    st.image(
        "https://i.pinimg.com/1200x/c6/dd/7d/c6dd7da295f066e26929a5232200df2a.jpg",
        width=100
    )
    st.markdown("## **Burnout Risk Analyzer**")
    
    page = st.selectbox(" Select Page", ["EDA", "Predict Burnout"])

    st.markdown("---")
    st.markdown("#### About")
    st.write(
        """
        This app helps identify and prevent workplace burnout using data analysis and 
        a machine learning model. It supports early detection and provides actionable 
        recommendations for at-risk individuals and promotes workplace well-being.
        """
    )

# Route to selected page
if page == "EDA":
    eda.run()
elif page == "Predict Burnout":
    prediction.run()
