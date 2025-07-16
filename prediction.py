import streamlit as st
import pickle
import json
import pandas as pd

# --- Load model and selected features ---
@st.cache_resource
def load_model():
    with open("model_pipeline_gb.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_features():
    with open("selected_features.json", "r") as f:
        return json.load(f)

def run():
    model = load_model()
    selected_features = load_features()

    # --- User Input ---
    job_role = st.selectbox("Job Role", ['Software Engineer', 'Data Analyst', 'HR', 'Marketing', 'Manager'])
    industry = st.selectbox("Industry", ['Technology', 'Healthcare', 'Finance', 'Education', 'Other'])
    years_exp = st.slider("Years of Experience", 0, 40, 3)
    work_location = st.selectbox("Work Location", ['Remote', 'On-site', 'Hybrid'])
    hours = st.slider("Hours Worked Per Week", 0, 100, 40)
    meetings = st.slider("Virtual Meetings per Week", 0, 20, 5)
    wlb_rating = st.slider("Work-Life Balance Rating (1 = Poor, 5 = Excellent)", 1, 5, 3)
    mental_access = st.selectbox("Access to Mental Health Resources", ['Yes', 'No'])
    isolation = st.slider("Social Isolation Rating (1 = Low, 5 = High)", 1, 5, 3)
    remote_sat = st.selectbox("Satisfaction with Remote Work", ['Satisfied', 'Neutral', 'Dissatisfied'])
    physical_activity = st.selectbox("Physical Activity", ['Low', 'Moderate', 'High'])

    # --- Format Data ---
    input_data = pd.DataFrame([{
        'Job_Role': job_role,
        'Industry': industry,
        'Years_of_Experience': years_exp,
        'Hours_Worked_Per_Week': hours,
        'Work_Location': work_location,
        'Number_of_Virtual_Meetings': meetings,
        'Work_Life_Balance_Rating': wlb_rating,
        'Access_to_Mental_Health_Resources': mental_access,
        'Social_Isolation_Rating': isolation,
        'Satisfaction_with_Remote_Work': remote_sat,
        'Physical_Activity': physical_activity
    }], columns=selected_features)

    # --- Inference ---
    if st.button("Predict Burnout Risk"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        st.write("Burnout Risk:", "🛑 At Risk" if prediction == 1 else "✅ Not at Risk")
        st.write("Risk Probability:", f"{round(probability * 100, 2)}%")

        if prediction == 1:
            st.markdown("### Recommendations for At-Risk Individuals")
            st.write("""
If you are at risk of burnout, seek support from a mental health professional. You may also find these helpful:
- Mindfulness & Recovery Guides: https://www.thisiscalmer.com/mindfulness-guides-and-ecourses
- Burnout Recovery Video: https://youtu.be/YyjBKqsJqAo?si=Rnhnq2S_Xyuujmya
- Healthline: https://www.healthline.com/health/mental-health/burnout-recovery
- BetterHelp (Online Therapy): https://www.betterhelp.com/
- Book an appointment with your in-house psychologist or HR.
""")
        else:
            st.markdown("### Preventive Recommendations")
            st.write("""
You're not currently at risk. To maintain well-being:
- Explore mindfulness tools: https://www.thisiscalmer.com/mindfulness-guides-and-ecourses
- Learn to prevent burnout: https://www.medicalnewstoday.com/articles/preventing-burnout
- Watch this awareness video: https://youtu.be/YyjBKqsJqAo?si=Rnhnq2S_Xyuujmya
""")

# --- Run the app ---
if __name__ == "__main__":
    run()
