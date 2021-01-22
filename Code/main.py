import streamlit as st
import pandas as pd
from sklearn import linear_model

import os

st.title("Heart Disease Prediction Application")
from PIL import Image


directory = os.path.dirname(__file__)
heartimage = os.path.join(directory, 'heartpredictiontoolimage.png')
agevsheart = os.path.join(directory, 'AgeVSHeartRate.png')
cholesterolage = os.path.join(directory, 'CholesterolAgeScatter.png')
restbpvsthalach = os.path.join(directory, 'RestBPvsThalach.png')

heartintro = Image.open(heartimage)
plot1 = Image.open(agevsheart)
plot2 = Image.open(cholesterolage)
plot3 = Image.open(restbpvsthalach)
st.image(heartintro, use_column_width= True)

pw = st.sidebar.text_input("Password")
if pw == 'nightowl':

        st.sidebar.header('Please select data for heart disease prediction ')

        def inputeddata():
            inputage = st.sidebar.slider('Age', 18, 100, 18)
            sex = st.sidebar.radio("Sex", ("Male", "Female"))
            chestpain = st.sidebar.radio("Chest Pain", ("Typical", "Asymptomatic", "Nonanginal", "Nontypical"))
            trestbps = st.sidebar.slider('Resting Blood Pressure',80,200,120)
            cholesterol = st.sidebar.slider('Cholesterol', 120,580,200)
            fbs = st.sidebar.radio("Fasting Blood Sugar Above 120 ml/dl",("Yes","No"))
            restecg = st.sidebar.radio("Resting ECG Results", ("Normal","Level 1","Level 2"))
            maxheartrate = st.sidebar.slider("Maximum heart rate",70,210,100)
            exangina = st.sidebar.radio("Exercise Induced Angina",("Yes","No"))
            oldpeak = st.sidebar.slider("ST Depression",0,6,0)
            slope = st.sidebar.slider("ST Depression Slope",0,2,0)
            caheart = st.sidebar.slider("Number of major vessels colored by fluoroscopy",0,4,0)
            thal = st.sidebar.radio("Thalium Stress Test Result",("Normal","Fixed Defect","Reversible Defect"))

            if thal == "Normal":
                thal = 1
            if thal == "Fixed Defect":
                thal = 2
            if thal == "Reversible Defect":
                thal = 3
            if exangina == "Yes":
                exangina = 1
            if exangina == "No":
                exangina = 0
            if restecg == "Normal":
                restecg = 0
            if restecg == "Level 1":
                restecg = 1
            if restecg == "Level 2":
                restecg = 2

            if fbs == "Yes":
                fbs = 1
            if fbs == "No":
                fbs = 0
            if chestpain == 'Typical':
                chestpain = 3
            if chestpain == 'Asymptomatic':
                chestpain = 0
            if chestpain == 'Nonanginal':
                chestpain = 2
            if chestpain == 'Nontypical':
                chestpain = 1

            if sex == 'Male':
                sex = 1
            if sex == 'Female':
                sex = 0

            patdata = {'Age': inputage,
                    'Sex': sex,
                    'Chest Pain': chestpain,
                    'Resting BP': trestbps,
                    'Cholesterol': cholesterol,
                    'FBS': fbs,
                    'Rest ECG': restecg,
                    'Thalach': maxheartrate,
                    'Exang': exangina,
                    'Oldpeak': oldpeak,
                    'Slope': slope,
                    'Ca': caheart,
                    'Thal': thal

                    }
            patientinfo = pd.DataFrame(patdata, index=[0])
            return patientinfo

        heartdataframe = inputeddata()

        st.subheader('Patient Data')
        st.write(heartdataframe)

        loadup = pd.read_csv(os.path.join(directory,'heart.csv'))



        heart = loadup
        X = heart[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
        Y = heart[['target']]



        logregression = linear_model.LogisticRegression(random_state= 1, max_iter=300)
        logreg = logregression
        logreg.fit(X,Y)


        score = logreg.score(X,Y)
        percentscore = score * 100
        roundedscore = round(percentscore,2)
        scoretodisplay = "The model has an accuracy of " + str(roundedscore) + "%"
        print(scoretodisplay)


        predict = logreg.predict(heartdataframe)

        st.subheader('Heart Disease Prediction')
        predictprobability = logreg.predict_proba(heartdataframe)
        testtext = "yes"
        if predict[0] == 0:
            testtext = "Predicted No Heart Disease From Patient Data"
        if predict[0] == 1:
            testtext = "Predicted Heart Disease From Patient Data"
        st.write(testtext)
        st.subheader('Prediction Probability Rating')
        st.write("0 : No Heart Disease")
        st.write("1 : Heart Disease")
        st.write(predictprobability)

        st.subheader("Logistic Regression Model Accuracy Rating")
        st.write(scoretodisplay)

        st.subheader("Scatter Plots From Previous Data Using K Means Clustering")
        st.image(plot1, use_column_width= True)
        st.image(plot2,use_column_width= True)
        st.image(plot3, use_column_width= True)