Lung Cancer Risk Prediction App

This project is a small machine learning application that predicts the risk level of lung cancer based on user-provided health inputs. It uses a trained classification pipeline that processes both numerical and categorical features to estimate risk as Low, Medium, or High.

The model was trained on a structured dataset containing lifestyle habits, symptoms, and basic demographic details. A preprocessing pipeline was used to clean the data, encode categorical variables, scale numerical inputs, and train a final classifier. Everything is bundled as a single scikit-learn pipeline for easier deployment.

The app is built with Streamlit and is fully interactive. You can try it here:

👉 Live App:
https://lung-cancer-predict-jxauwbvnma2kyu3ccgwi7u.streamlit.app/

Features

User-friendly form for entering all the model features

Clean UI showing:

Predicted lung cancer risk

Prediction confidence

Confusion matrix of the model

Preprocessing and model packed inside one pipeline

Completely deployable through Streamlit Cloud

How the Model Works

Reads and preprocesses the dataset

Encodes categorical fields using one-hot encoding

Scales numerical values

Trains a multiclass classifier

Saves everything as a single pipeline file (lung_cancer_pipeline.pkl)

This ensures the Streamlit app only needs to load one file to run predictions.

Tech Stack

Python

Scikit-learn

Pandas / NumPy

Matplotlib

Streamlit