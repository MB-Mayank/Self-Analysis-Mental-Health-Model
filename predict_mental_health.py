import pandas as pd
import os
import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from category_encoders import TargetEncoder
import streamlit as st
import google.generativeai as genai

import warnings
warnings.filterwarnings('ignore')

class RenameColumnsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        df = df.copy()
        
        # Renaming columns
        df.rename(columns={
            "Have_you_ever_had_suicidal_thoughts_?": "Suicidal_Thoughts",
            "Family_History_of_Mental_Illness": "Family_History",
        }, inplace=True)

        return df
class TargetEncodeCategoricalVariablesWithCVTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, n_splits=5, random_state=42):
        if cols is None:
            self.cols = ['Profession', 'Degree', 'City']
        else:
            self.cols = cols
        self.n_splits = n_splits
        self.random_state = random_state
        self.target_encoder = TargetEncoder(cols=self.cols)

    def fit(self, df, y=None):
        if 'Depression' in df.columns:
            self.df = df.copy()
            self.y = y
        return self

    def transform(self, df):
        df = df.copy()
        
        if 'Depression' in df.columns:
            # Initializing cross-validation
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

            for train_index, val_index in kf.split(self.df, self.df['Depression']):
                train_fold = self.df.iloc[train_index]
                val_fold = self.df.iloc[val_index]

                # Applying TargetEncoder to columns for training and validation
                self.target_encoder.fit(train_fold[self.cols], train_fold['Depression'])
                val_fold_encoded = self.target_encoder.transform(val_fold[self.cols])

                # Updating the transformed columns in the validation set
                df.loc[val_index, self.cols] = val_fold_encoded
        else:
            # Applying TargetEncoder directly without cross-validation
            df[self.cols] = self.target_encoder.transform(df[self.cols])
        
        return df
    # Encoding Categorical Variables
class EncodeCategoricalVariablesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}

    def fit(self, df, y=None):
        # Initialize LabelEncoders for binary columns only
        self.binary_columns = ['Family_History', 'Suicidal_Thoughts', 'Gender',
                               'Working_Professional_or_Student']
        for col in self.binary_columns:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].fillna('-1').astype(str))
                
                self.label_encoders[col] = le
        return self

    def transform(self, df):
        df = df.copy()

        # Apply LabelEncoder to binary columns
        for col in self.binary_columns:
            if col in df.columns:
                le = self.label_encoders.get(col)
                if le is not None:
                    df[col] = le.transform(df[col].fillna('-1').astype(str))


        diet_mapping = {
            'More Healty': 0,
            'Healthy': 1,
            'Less than Healthy': 2,
            'Less Healthy': 2,
            'Moderate': 3,
            'Unhealthy': 4,
            'No Healthy': 4,
        }
        if "Dietary_Habits" in df.columns:
            df['Dietary_Habits'] = df['Dietary_Habits'].map(diet_mapping)
            df['Dietary_Habits'] = df['Dietary_Habits'].fillna(
                df['Dietary_Habits'].mode().iloc[0]
            )
            
        return df
    

class ApplyPressureLogicTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        df = df.copy()
        
        df['Academic_Pressure'] = df.apply(
            lambda row: np.nan if row['Working_Professional_or_Student'] == 1
            else row['Academic_Pressure'], axis=1
        )
        
        df['Work_Pressure'] = df.apply(
            lambda row: np.nan if row['Working_Professional_or_Student'] == 0
            else row['Work_Pressure'], axis=1
        )
        
        return df
    
    # Creating Derived Columns
class CreateDerivedColumnsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        df = df.copy()

        if 'Gender' in df.columns and 'Work_Pressure' in df.columns:
            gender_work_pressure_mean = df.groupby('Gender')['Work_Pressure'].mean().to_dict()
            df['Gender_Work_Pressure'] = df['Gender'].map(gender_work_pressure_mean)

        if 'Suicidal_Thoughts' in df.columns and 'Job_Satisfaction' in df.columns:
            suicidal_job_satisfaction_mean = df.groupby('Suicidal_Thoughts')['Job_Satisfaction'].mean().to_dict()
            df['Suicidal_Job_Satisfaction'] = df['Suicidal_Thoughts'].map(suicidal_job_satisfaction_mean)

        if 'Financial_Stress' in df.columns and 'Academic_Pressure' in df.columns:
            financial_academic_pressure_mean = df.groupby('Financial_Stress')['Academic_Pressure'].mean().to_dict()
            df['Financial_Academic_Pressure'] = df['Financial_Stress'].map(financial_academic_pressure_mean)

        if 'Financial_Stress' in df.columns and 'Study_Satisfaction' in df.columns:
            financial_study_satisfaction_mean = df.groupby('Financial_Stress')['Study_Satisfaction'].mean().to_dict()
            df['Financial_Study_Satisfaction'] = df['Financial_Stress'].map(financial_study_satisfaction_mean)

        return df
    
class ApplyWorkStudyHoursLogicTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
            pass
        
    def fit(self, X, y=None):

            return self
        
    def transform(self, df):
            
            df = df.copy()

            df['Work_Hours'] = df.apply(
                lambda row: np.nan if pd.isna(row['Work/Study_Hours']) 
                else row['Work/Study_Hours'] if row['Working_Professional_or_Student'] == 1
                else 0,
                axis=1
            )
            
            df['Study_Hours'] = df.apply(
                lambda row: np.nan if pd.isna(row['Work/Study_Hours']) 
                else row['Work/Study_Hours'] if row['Working_Professional_or_Student'] == 0
                else 0,
                axis=1
            )
            
            df.drop(['Work/Study_Hours'], axis=1, inplace=True)
            
            return df
    
    degree = {
    "BCom": "B.Com", "B.Com": "B.Com", "B.Comm": "B.Com",
    "B.Tech": "B.Tech", "BTech": "B.Tech", "B.T": "B.Tech",
    "BSc": "B.Sc", "B.Sc": "B.Sc", "Bachelor of Science": "B.Sc",
    "BArch": "B.Arch", "B.Arch": "B.Arch",
    "BA": "B.A", "B.A": "B.A",
    "BBA": "BBA", "BB": "BBA",
    "BCA": "BCA",
    "BE": "BE",
    "BEd": "B.Ed", "B.Ed": "B.Ed",
    "BPharm": "B.Pharm", "B.Pharm": "B.Pharm",
    "BHM": "BHM",
    "LLB": "LLB", "LL B": "LLB", "LL BA": "LLB", "LL.Com": "LLB", "LLCom": "LLB",
    "MCom": "M.Com", "M.Com": "M.Com",
    "M.Tech": "M.Tech", "MTech": "M.Tech", "M.T": "M.Tech",
    "MSc": "M.Sc", "M.Sc": "M.Sc", "Master of Science": "M.Sc",
    "MBA": "MBA",
    "MCA": "MCA",
    "MD": "MD",
    "ME": "ME",
    "MEd": "M.Ed", "M.Ed": "M.Ed",
    "MArch": "M.Arch", "M.Arch": "M.Arch",
    "MPharm": "M.Pharm", "M.Pharm": "M.Pharm",
    "MA": "MA", "M.A": "MA",
    "MPA": "MPA",
    "LLM": "LLM",
    "PhD": "PhD",
    "MBBS": "MBBS",
    "CA": "CA",
    "Class 12": "Class 12", "12th": "Class 12",
    "Class 11": "Class 11", "11th": "Class 11"
}

# Sleep Duration Standardization
class StandardizeSleepDurationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        df = df.copy()
        sleep_mapping = {
            'More than 8 hours': 9,  
            'Less than 5 hours': 4,  
            '5-6 hours': 5.5,
            '6-7 hours': 6.5,
            '7-8 hours': 7.5,
            '8-9 hours': 8.5,
            '9-11 hours': 10, 
            '10-11 hours': 10.5,
            '4-6 hours': 5,
            '6-8 hours': 7,
            '1-2 hours': 1.5,
            '2-3 hours': 2.5,
            '3-4 hours': 3.5,
            '4-5 hours': 4.5,
            '8 hours' : 8,
            '1-3 hours': 2,
            '3-6 hours' : 4.5,
            '9-6 hours' : 7.5
        }
        if "Sleep_Duration" in df.columns:
            df['Sleep_Duration'] = df['Sleep_Duration'].map(sleep_mapping)
            df['Sleep_Duration'] = df['Sleep_Duration'].fillna(df['Sleep_Duration'].median())
            
        return df

class FillMissingValuesWithMedianTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians = {}
        
    def fit(self, X, y=None):
        # Calculates the median for all numeric columns with missing values
        self.medians = X.median(numeric_only=True).to_dict()
        return self
    
    def transform(self, df):
        df = df.copy()
        
        # Fills in missing values ​​with calculated medians
        for col, median in self.medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)
        
        return df 
with open("train_transformed.pkl", "rb") as f:
    train_df = pickle.load(f)
with open("my_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

loaded_model = joblib.load("best_xgb_model.pkl")
genai.configure(api_key="AIzaSyDq5MG8mkJxVW2Dv1CCDjJzyqUzUb_UjJU")

gemini_model = genai.GenerativeModel("gemini-pro")
# Define the input fields
required_columns = [
    "id", "Name", "Gender", "Age", "City", "Working Professional or Student",
    "Profession", "Academic Pressure", "Work Pressure", "CGPA",
    "Study Satisfaction", "Job Satisfaction", "Sleep Duration",
    "Dietary Habits", "Degree", "Have you ever had suicidal thoughts ?",
    "Work/Study Hours", "Financial Stress", "Family History of Mental Illness"
]

# Streamlit UI
def main():
    st.title("Self-Analysis Mental Health Model")
    
    user_inputs = {}
    
    with st.form("user_form"):
        for column in required_columns:
            if column in ["Age", "CGPA", "Work/Study Hours", "Academic Pressure", "Work Pressure", "Financial Stress"]:
                user_inputs[column] = st.number_input(f"Enter {column}")
            elif column == "id":
                user_inputs[column] = st.number_input(f"Enter {column}")
            else:
                user_inputs[column] = st.text_input(f"Enter {column}")
        
        submit_button = st.form_submit_button("Save & Predict")
    
    if submit_button:
        save_to_csv(user_inputs)
        prediction = make_prediction()
        results = []
        for i, prediction in enumerate(prediction):
            prediction_label = "Depressed" if prediction == 1 else "Not Depressed"
            result_text = f"{user_inputs['Name']} (ID: {user_inputs['id']}) is {prediction_label}"
            results.append(result_text)
            
            if prediction == 1:  # If depressed, generate insights using Gemini
                explanation = generate_reason_suggestions(user_inputs)
                results.append(f"Reason & Suggestion: {explanation}")
        
        for result in results:
            st.success(result)


def save_to_csv(user_data, filename="user_input1.csv"):
    """Saves the user input to a CSV file."""
    df = pd.DataFrame([user_data])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)


def make_prediction():
    """Transforms user input and makes a prediction using the trained model."""
    df = pd.read_csv("user_input1.csv")
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df[['Profession', 'Degree']] = df[['Profession', 'Degree']].astype(str)
    train_df = pipeline.transform(df)
    
    
    
    train_df.drop(['id'],axis = 1,inplace = True)
    
    train_df.drop(['Name'],axis = 1,inplace = True)
    loaded_model = joblib.load("best_xgb_model.pkl")
    predictions = loaded_model.predict(train_df)
    return predictions
def generate_reason_suggestions(user_data):
    """Uses Gemini LLM to analyze user data and provide explanations & suggestions."""
    prompt = f"""
    Analyze the following mental health data and determine potential reasons for depression along with suggestions:
    {user_data}
    Provide a short, meaningful response.
    """
    response = gemini_model.generate_content(prompt)
    return response.text if response else "No explanation available."

if __name__ == "__main__":
    main()
