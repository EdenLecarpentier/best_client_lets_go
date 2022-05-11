import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from pandas import get_dummies
import pandas as pd
from imblearn.pipeline import Pipeline as imbpipeline
import seaborn as sns
from sklearn.pipeline import Pipeline , make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

st.write("""
# Penguin Prediction App
This app predicts the **Palmer Penguin** species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")


df  = pd.read_csv("application_train.csv")
df

df = df.dropna()
df

X = df.drop(["TARGET" ], axis=1)
y = df.TARGET
le = LabelEncoder()

categ = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY' , 'NAME_CONTRACT_TYPE' ,'NAME_TYPE_SUITE' , 'NAME_INCOME_TYPE' , 'NAME_EDUCATION_TYPE' , 'NAME_FAMILY_STATUS' , 'NAME_HOUSING_TYPE' , 'WEEKDAY_APPR_PROCESS_START' , 'OCCUPATION_TYPE' , 'ORGANIZATION_TYPE' , 'FONDKAPREMONT_MODE' , 'HOUSETYPE_MODE' , 'WALLSMATERIAL_MODE' ,'EMERGENCYSTATE_MODE' ]
df[categ] = df[categ].apply(le.fit_transform)

df

X = df.drop(["TARGET"] , axis=1)
y = df.TARGET

X_train , X_test , y_train , y_test = train_test_split(X , y , random_state=42 , test_size=0.2)

preprocessing_ss = Pipeline(steps=[
    ('normal' , StandardScaler())])
sm = SMOTE(sampling_strategy='auto')
pipeline_xgb_sm = imbpipeline(steps = [['SMOTE', sm],
                                    ['classifier', XGBClassifier()]
                                     ])
pipeline_xgb_sm.fit(X_train , y_train)

y_pred_xgb_sm = pipeline_xgb_sm.predict(X_test)

load_clf = pickle.load(open('test_pickle.pkl', 'rb'))