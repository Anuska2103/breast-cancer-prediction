import streamlit as st
import pandas as pd
import numpy as np
import pickle
with open('model1.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the trained model

def predict_diagnosis(features):
    return model.predict([features])[0]

# Streamlit app layout
st.title('Breast Cancer Prediction App')
st.markdown('please enter the following specification in order to get the prediction')


feature1 = st.number_input('texture mean', value=0.0)
feature2 = st.number_input('smoothness mean', value=0.0)
feature3 = st.number_input('compactness mean', value=0.0)
            
feature4 = st.number_input('concave points mean', value=0.0)
feature5 = st.number_input('symmetry mean', value=0.0)
feature6 = st.number_input('fractal dimension mean', value=0.0)
feature7= st.number_input('fractal dimension worst', value=0.0)
feature8= st.number_input('texture', value=0.0)
feature9= st.number_input('area', value=0.0)
feature10 = st.number_input('smoothness', value=0.0)
feature11= st.number_input('compactness', value=0.0)
feature12= st.number_input('concavity', value=0.0)
feature13= st.number_input('concave points', value=0.0)
feature14= st.number_input('symmetry', value=0.0)
feature15= st.number_input('fractal dimension', value=0.0)
feature16= st.number_input('texture_worst', value=0.0)
feature17= st.number_input('area worst	', value=0.0)
feature18= st.number_input('smoothness worst', value=0.0)
feature19= st.number_input('compactness worst', value=0.0)
feature20= st.number_input('concavity worst', value=0.0)
feature21= st.number_input('concave points worst', value=0.0)
feature22= st.number_input('symmetry worst', value=0.0)
feature23= st.number_input('unamed', value=0.0)


# Add more input fields based on the number of features used in your model

# Prepare input data for prediction
features = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19, feature20, feature21, feature22, feature23,]  # Adjust based on your features

# Predict and display result
if st.button('Predict'):
    prediction = predict_diagnosis(features)
    diagnosis = 'the cells found to be Malignant(cancerous)' if prediction == 1 else 'the cells are Benign(non-      cancerous)'
    st.write(f'Prediction: {diagnosis}')
