import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import os

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    print("Cleaning data...")
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

def encode_features(df):
    print("Encoding features...")
    df = df.copy()
    
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                        'Contract', 'PaymentMethod']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def scale_features(df):
    print("Scaling features...")
    df = df.copy()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def handle_imbalance(X, y):
    print("Handling imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def run_preprocessing(input_path, output_path):
    df = load_data(input_path)
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_res, y_train_res = handle_imbalance(X_train, y_train)
    
    train_df = pd.concat([pd.DataFrame(X_train_res, columns=X.columns), pd.Series(y_train_res, name='Churn')], axis=1)
    test_df = pd.concat([pd.DataFrame(X_test, columns=X.columns).reset_index(drop=True), pd.Series(y_test, name='Churn').reset_index(drop=True)], axis=1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving preprocessed data to {output_path}...")
    full_processed_df = pd.concat([train_df, test_df], axis=0)
    full_processed_df.to_csv(output_path, index=False)
    print("Preprocessing complete!")

if __name__ == '__main__':
    # Path input dan output disesuaikan dengan struktur folder di screenshot
    input_file = '../WA_Fn-UseC_-Telco-Customer-Churn_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    output_file = 'WA_Fn-UseC_-Telco-Customer-Churn_preprocessing/telco_churn_clean.csv'
    
    if os.path.exists(input_file):
        run_preprocessing(input_file, output_file)
    else:
        print(f"File not found: {input_file}. Please ensure raw dataset is placed in the correct folder.")
