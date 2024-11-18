import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from PyPDF2 import PdfReader

# Load the trained model and TfidfVectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Function to clean the resume text
def cleanResume(resumeText):
    resumeText = re.sub('http\\S+\\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[0-9]+', '', resumeText)  # remove numbers
    resumeText = re.sub(r'\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

# Streamlit app title and description
st.markdown("<span style='color:blue; font-size: 36px'>Resume Parser Web App</span>", unsafe_allow_html=True)
st.markdown("<span style='font-size:18px'>Developed by Aditya Kumar</span>", unsafe_allow_html=True)
# Image
st.image("pic.png", use_column_width=True, width=200)  # reduce the size of the picture

# File uploader for resume (PDF and text files)
uploaded_file = st.file_uploader("Choose a resume file", type=["txt", "pdf"])

if uploaded_file is not None:
    # Check if the uploaded file is a PDF
    if uploaded_file.type == "application/pdf":
        # Read the uploaded PDF file
        pdf_file = PdfReader(uploaded_file)
        text = ""
        for page in range(len(pdf_file.pages)):
            text += pdf_file.pages[page].extract_text()
    else:
        # The uploaded file is a text file, so read it
        text = uploaded_file.read().decode('utf-8')

    # Clean the uploaded resume
    cleaned_resume = cleanResume(text)

    # Transform the cleaned resume using the trained TfidfVectorizer
    input_features = tfidf.transform([cleaned_resume])

    # Make the prediction using the loaded classifier
    prediction_id = clf.predict(input_features)[0]

    # Map prediction ID to category name
    category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Designing",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
        6: "Data Science",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts",
        7: "Database",
        11: "Electrical Engineering",
        14: "Health and fitness",
        19: "PMO",
        4: "Business Analyst",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Security Engineer",
        21: "SAP Developer",
        5: "Civil Engineer",
        0: "Advocate",
    }

    # Display the predicted job category
    category_name = category_mapping.get(prediction_id, "Unknown")
    st.subheader(f"Predicted Category: {category_name}")

    # Display the skills in the predicted category
    skills_mapping = {
    "Data Science": ["Python", "R", "SQL", "Machine Learning", "Deep Learning", "Data Visualization", "Statistics"],
    "Java Developer": ["Java", "Spring", "Hibernate", "MySQL", "Oracle"],
    "Python Developer": ["Python", "Django", "Flask", "NumPy", "Pandas"],
    "Testing": ["Selenium", "JUnit", "TestNG", "Manual Testing", "Automation Testing"],
    "DevOps Engineer": ["Docker", "Kubernetes", "Jenkins", "Ansible", "Cloud Computing"],
    "Web Designing": ["HTML", "CSS", "JavaScript", "React", "Angular", "Vue.js"],
    "HR": ["Recruitment", "Talent Management", "Employee Engagement", "Benefits Administration", "Payroll"],
    "Hadoop": ["HDFS", "MapReduce", "Hive", "Pig", "Spark"],
    "Blockchain": ["Bitcoin", "Ethereum", "Smart Contracts", "Blockchain Development", "Cryptocurrency"],
    "ETL Developer": ["Informatica", "Talend", "Microsoft SSIS", "Data Warehousing", "Data Integration"],
    "Operations Manager": ["Project Management", "Supply Chain Management", "Inventory Management", "Logistics", "Operations Research"],
    "Sales": ["Salesforce", "CRM", "Sales Strategy", "Account Management", "Business Development"],
    "Mechanical Engineer": ["CAD Design", "Mechanical Systems", "Thermodynamics", "Materials Science", "Manufacturing"],
    "Arts": ["Graphic Design", "Digital Art", "Painting", "Sculpture", "Photography"],
    "Database": ["MySQL", "Oracle", "SQL Server", "MongoDB", "Database Administration"],
    "Electrical Engineering": ["Circuit Design", "Microcontrollers", "Electronics", "Power Systems", "Control Systems"],
    "Health and fitness": ["Personal Training", "Nutrition", "Wellness", "Fitness Coaching", "Health Education"],
    "PMO": ["Project Management", "Portfolio Management", "Program Management", "Risk Management", "Quality Assurance"],
    "Business Analyst": ["Business Process Improvement", "Requirements Gathering", "Data Analysis", "Solution Design", "Implementation"],
    "DotNet Developer": ["C#", "ASP.NET", "VB.NET", "ADO.NET", "Entity Framework"],
    "Automation Testing": ["Selenium", "Appium", "TestComplete", "Ranorex", "Automation Frameworks"],
    "Network Security Engineer": ["Firewalls", "VPN", "Intrusion Detection", "Penetration Testing", "Security Auditing"],
    "SAP Developer": ["ABAP", "SAP UI5", "SAP Fiori", "SAP HANA", "SAP BW"],
    "Civil Engineer": ["Structural Analysis", "Construction Management", "Transportation Engineering", "Water Resources", "Geotechnical Engineering"],
    "Advocate": ["Law", "Litigation", "Corporate Law", "Intellectual Property", "Contract Law"]
}
    skills = skills_mapping.get(category_name, [])
    st.subheader(f"Skills in {category_name}:")
    for skill in skills:
        st.write(skill)

# Add a section to describe the project and how to use it
st.header("About this Project")
st.write("This is a resume parser web app that predicts the job category based on the uploaded resume. Simply upload your resume in PDF or text format, and the app will predict the most likely job category and display the relevant skills.")
st.write("To use this app, follow these steps:")
st.write("1. Upload your resume file (PDF or text) using the file uploader above.")
st.write("2. Wait for the app to process the file and make a prediction.")
st.write("3. The predicted job category and relevant skills will be displayed below.")
st.write("Note: This app is trained on a dataset of resumes and may not always produce accurate results. Use at your own discretion.")