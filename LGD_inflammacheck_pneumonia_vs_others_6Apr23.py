# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:46:44 2023

@author: ldcruz
"""


# load and evaluate a saved model
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import streamlit as st
from sklearn.metrics import accuracy_score
from numpy import loadtxt
import pandas as pd

import pickle

filename = 'VICTORY_extratrees_pneumonia_vs_others_6Apr23.sav'  # specify the saved filename

# Restore all objects and model stored in tuple pickle file
ET_model, X_train_smote, y_train_smote, X_test_smote, y_test_smote, y_pred_class_on_test = pickle.load(
    open(filename, 'rb'))


print("Accuracy:", metrics.accuracy_score(y_test_smote, y_pred_class_on_test))


# Get the feature input from the user

def get_user_input():

    Biologic = st.sidebar.slider('Biologic', 0, 1)
    H2O2 = st.sidebar.slider("H2O2 [uM]", 0.00, 3.00, 0.60)
    Peak_CO2 = st.sidebar.slider("Peak CO2 [%]", 1.00, 25.00, 5.00)
    EXH_flowrate = st.sidebar.slider("EXH flow rate [L/min]", 0.00, 30.00, 8.93)
    Peak_Breath_temp = st.sidebar.slider("Peak breath temperature [oC]", 0.00, 32.00, 27.68)
    Peak_Breath_humidity = st.sidebar.slider("Peak breath humidity", 5.00, 100.00, 64.86)
    Ambient_temperature = st.sidebar.slider("Ambient temperature [oC]", -10.00, 38.00, 25.00)
    Ambient_humidity = st.sidebar.slider("Ambient humidity[%]", 0.00, 90.00, 40.00)
    Exhale_time = st.sidebar.slider("EXHALE time[s]", 10.00, 250.00, 75.00)


    # Store a dictionary into a variable
    user_data = {'Biologic': Biologic,
                 "H2O2 [uM]": H2O2,
                 "Peak CO2 [%]": Peak_CO2,
                 "EXH flow rate [L/min]": EXH_flowrate,
                 "Peak breath temperature [oC]": Peak_Breath_temp,
                 "Peak breath humidity": Peak_Breath_humidity,
                 "Ambient temperature [oC]": Ambient_temperature,
                 "Ambient humidity[%]": Ambient_humidity,
                 "EXHALE time[s]":Exhale_time
                 }

    # Transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features


# store the user's input into a variable
user_input = get_user_input()

# set a subheader and display the users input
#st.subheader('User input:')
# so when user puts in a value, we can see it on the app
#st.write(user_input)

st.header('INFLAMMACHECK model for pneumonia vs no pneumonia')
st.subheader('Machine learning model - ExtraTrees for prediction of pneumonia using INFLAMMACHECK')

st.write(' ')
st.write('A collaborative project between Portsmouth University Hospital, Exhalation Technology Ltd.')
st.write(' ')
st.write(' ')
st.write('**probable clinical risk of PNEUMONIA >76% accuracy suggests:**')
y_new = ET_model.predict(user_input)
#st.write(y_new)

if y_new == 0:
    st.write("**within normal parameters**")
    
elif y_new == 1:
    st.write("**needs respiratory assessment but no Pneumonia**")
    
elif y_new == 2:
    st.write("**PROBABLE CLINICAL RISK of PNEUMONIA**")

st.write('')
st.write('Note: this model is in development stage, not yet for approved clinical use')