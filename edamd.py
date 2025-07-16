import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run():
  st.title("Workplace Burnout Prediction EDA")

  # HEADER GIF
  st.markdown("""
  <div style='text-align: center;'>
    <img src='https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExb3cwbjZvcHB4N3F6ZTFobmN4NWZia2RxcW04ZXgwcmxxMWtqMXNyNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4ytUZzb1pRPBS/giphy.gif' width='400'>
  </div>
  """, unsafe_allow_html=True)

  # DESCRIPTION
  st.write("### Description")
  st.write("""
  Workplace burnout is a serious issue caused by prolonged stress, leading to exhaustion, reduced performance, and detachment. 
  It is recognized by the World Health Organization as an occupational phenomenon. Burnout affects both individuals and organizations, causing anxiety, absenteeism, and high turnover, which can lead to major financial losses. 
  This analysis explores key factors like stress, sleep, workload, and productivity to help identify potential burnout risks and inform prevention strategies.
  """)

  # LOAD DATA
  df1 = pd.read_csv('Impact_of_Remote_Work_on_Mental_Health.csv')

  # SECTION 4.2
  st.header("Distribution of Each Factor")

  # 4.2.1. Working Hours
  st.subheader("Distribution of Working Hours")
  bins = [0, 20, 30, 40, 50, 60, 70, 100]
  labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
  df1['Working_Hours_Group'] = pd.cut(df1['Hours_Worked_Per_Week'], bins=bins, labels=labels, right=False)
  hours_count = df1['Working_Hours_Group'].value_counts().sort_index()
  fig1, ax1 = plt.subplots()
  sns.barplot(x=hours_count.index, y=hours_count.values, palette='pastel', ax=ax1)
  ax1.set_title('Distribution of Working Hours per Week')
  ax1.set_xlabel('Working Hour Group')
  ax1.set_ylabel('Number of Employees')
  st.pyplot(fig1)
  st.markdown("Most employees fall within the 21–50 hour range, with many working over 51 hours, approaching the WHO’s risk threshold.")

  # 4.2.2. Stress Level
  st.subheader("Distribution of Stress Level")
  stress_counts = df1['Stress_Level'].value_counts().reindex(['Low', 'Medium', 'High'])
  fig2, ax2 = plt.subplots()
  ax2.pie(stress_counts, labels=stress_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=140)
  ax2.set_title('Distribution of Stress Levels')
  st.pyplot(fig2)
  st.markdown("Stress is evenly distributed, with a slight tilt toward high stress. It shows burnout risk is spread across the workforce.")

  # 4.2.3. Sleep Quality
  st.subheader("Distribution of Sleep Quality")
  fig3, ax3 = plt.subplots()
  sns.countplot(x='Sleep_Quality', data=df1, order=['Poor', 'Average', 'Good'], palette='pastel', ax=ax3)
  ax3.set_title("Distribution of Sleep Quality")
  st.pyplot(fig3)
  st.markdown("A significant number report poor sleep—an important burnout indicator requiring intervention.")

  # 4.2.4. Productivity Change
  st.subheader("Distribution of Productivity Change")
  fig4, ax4 = plt.subplots()
  sns.countplot(x='Productivity_Change', data=df1, palette='coolwarm', ax=ax4)
  ax4.set_title("Distribution of Productivity Change")
  st.pyplot(fig4)
  st.markdown("Decreased productivity is the most common, highlighting signs of burnout such as reduced focus or motivation.")

  # SECTION 4.3
  st.header("Analysis")

  # 4.3.1. Proportion of At-Risk Employees
  st.subheader("Proportion of At-Risk Employees")
  df1['At_Risk'] = (
    (df1['Stress_Level'] == 'High').astype(int) +
    (df1['Sleep_Quality'] == 'Poor').astype(int) +
    (df1['Hours_Worked_Per_Week'] > 50).astype(int) +
    (df1['Productivity_Change'] == 'Decrease').astype(int)
  ) >= 2
  risk_counts = df1['At_Risk'].value_counts().sort_index()
  labels = ['Not At Risk', 'At Risk']
  colors = ['lightblue', 'salmon']
  fig5, ax5 = plt.subplots()
  ax5.pie(risk_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
  ax5.set_title('Proportion of At-Risk Employees')
  st.pyplot(fig5)

  # 4.Overworked Job Role
  st.subheader("Overworked Job Role")
  overworked = df1[df1['Hours_Worked_Per_Week'] > 55]
  fig6, ax6 = plt.subplots()
  sns.countplot(x='Job_Role', data=overworked, ax=ax6)
  ax6.set_title("Job Role of Employees Working > 55 Hours/Week")
  ax6.tick_params(axis='x', rotation=45)
  st.pyplot(fig6)

  # 4.3.3. Job Role With High Stress
  st.subheader("Job Role With High Stress")
  high_stress = df1[df1['Stress_Level'] == 'High']
  fig7, ax7 = plt.subplots()
  sns.countplot(x='Job_Role', data=high_stress, order=high_stress['Job_Role'].value_counts().index, palette='Reds', ax=ax7)
  ax7.set_title('High Stress Counts by Job Role')
  ax7.tick_params(axis='x', rotation=45)
  st.pyplot(fig7)

  # 4.3.4. Stress Level of Overworked Employees
  st.subheader("Stress Level of Overworked Employees")
  fig8, ax8 = plt.subplots()
  sns.countplot(x='Stress_Level', data=overworked, order=['Low', 'Medium', 'High'], palette='Reds', ax=ax8)
  ax8.set_title("Stress Level of Employees Working > 55 Hours/Week")
  st.pyplot(fig8)
