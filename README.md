# Self-Analysis Mental Health Model

## 📌 Project Overview
The **Self-Analysis Mental Health Model** is an AI-powered system that predicts an individual's likelihood of experiencing depression based on demographic, academic/work, lifestyle, and mental health indicators. The system utilizes data preprocessing, feature engineering, machine learning classification, and a generative AI model for insights and suggestions.

## 🚀 Technologies Used
- **Programming Language**
- **Machine Learning Libraries**
- **Model Deployment & UI**: Streamlit
- **Data Processing & Feature Engineering**:
  - Label Encoding
  - Target Encoding with Cross-Validation
  - Handling Missing Values
  - Feature Transformations (Standardizing Sleep Duration, Work/Study Hours Logic, etc.)
- **AI Model for Insights**: Google Gemini AI
- **File Handling**: Joblib, Pickle, CSV

## 📊 Dataset Format
The input dataset consists of user records with the following structure:

| ID | Name | Gender | Age | City | Professional Status | Profession | Academic Pressure | Work Pressure | CGPA | Study Satisfaction | Job Satisfaction | Sleep Duration | Dietary Habits | Degree | Suicidal Thoughts | Work/Study Hours | Financial Stress | Family History of Mental Illness |
|----|------|--------|-----|------|--------------------|------------|-------------------|--------------|------|------------------|-----------------|---------------|--------------|--------|----------------|----------------|----------------|------------------------------|
| 140703 | Nalini | Female | 23.0 | Rajkot | Student | - | 5.0 | - | 6.84 | 1.0 | - | More than 8 hours | Moderate | BSc | Yes | 10.0 | 4.0 | No |
| 140704 | Shaurya | Male | 47.0 | Kalyan | Working Professional | Teacher | - | 5.0 | - | 5.0 | 7-8 hours | Moderate | BCA | Yes | 3.0 | 4.0 | No |

## 🔧 Data Preprocessing Pipeline
### ✏️ Column Renaming
- Standardizes column names for consistency.
- Example: `Family_History_of_Mental_Illness` → `Family_History`.

### 🔄 Encoding Categorical Variables
- Label Encoding for binary features like `Gender`, `Family_History`, `Suicidal_Thoughts`.
- Target Encoding for multi-category features (`Profession`, `Degree`, `City`) using cross-validation.

### 🏗️ Handling Missing Values
- Median imputation for numerical features.
- Mode imputation for categorical features.

### 🏋️ Feature Engineering
- **Derived Columns**:
  - `Gender_Work_Pressure`: Average `Work_Pressure` grouped by `Gender`.
  - `Suicidal_Job_Satisfaction`: Mean `Job_Satisfaction` for individuals with `Suicidal_Thoughts`.
  - `Financial_Academic_Pressure`: Relationship between `Financial_Stress` and `Academic_Pressure`.
- **Work/Study Hours Logic**:
  - If `Professional_Status = Working Professional`, `Study_Hours = 0`, `Work_Hours = Work/Study_Hours`.
  - If `Professional_Status = Student`, `Work_Hours = 0`, `Study_Hours = Work/Study_Hours`.
- **Sleep Duration Standardization**:
  - Converts sleep duration categories into numerical values (e.g., `More than 8 hours` → `9`).

## 🎯 Model Training
### 🧠 Machine Learning Model
- **Algorithm Used**: XGBoost (Trained using `best_xgb_model.pkl`)
- **Training Process**:
  - Feature-engineered data is used for supervised learning.
  - Target variable: `Depression` (binary classification: `Depressed` or `Not Depressed`).
  - Cross-validation used for model optimization.

## 🛠️ Setup Instructions
### 1️⃣ Clone the Repository
```bash
git clone <repository_url>
cd <repository_name>
```

### 2️⃣ Set Up Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate  # For Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download Pre-trained Models
Ensure the following files are present:
- `best_xgb_model.pkl`
- `my_pipeline.pkl`
- `train_transformed.pkl`

### 5️⃣ Set Up API Key for Google Gemini AI
Update `predict_mental_health.py`:
```python
genai.configure(api_key="your_google_gemini_api_key")
```

### 6️⃣ Run the Application and give the imput based(means type ) on above tabe only for now
```bash
streamlit run predict_mental_health.py
```

## 🔍 How It Works
1. User fills out the Streamlit form and submits.
2. Data is saved to `user_input1.csv`.
3. Preprocessing pipeline transforms the input.
4. XGBoost model predicts mental health status.
5. If `Depressed`, Gemini AI generates explanations & recommendations.
6. Results are displayed to the user.

## ✅ Conclusion
The **Self-Analysis Mental Health Model** provides a structured, data-driven approach to predicting depression risk based on lifestyle and mental health indicators. Using machine learning and generative AI, the system not only classifies mental health conditions but also provides meaningful insights for self-improvement.

---

⭐ *Feel free to contribute by submitting issues or pull requests!*

