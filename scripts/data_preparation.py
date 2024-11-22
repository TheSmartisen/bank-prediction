import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# ### Configuration
DATA_PATH = "data/"
RAW_FILE = DATA_PATH + "raw/load_data.csv"
WITHOUT_MISSING_FILE = DATA_PATH + "processed/data_without_missing.csv"
FOR_TEST_FILE = DATA_PATH + "processed/data_for_test.csv"
CLEAN_FOR_TRAINING = DATA_PATH + "processed/data_clean_for_training.csv"
PICKLE_PATH = "outputs/pickle/"
ENCODER_FILE = PICKLE_PATH + "encoders_config.pkl"

# Charger les données brutes
df = pd.read_csv(filepath_or_buffer=RAW_FILE, sep=",", header=0)

# ### Gestion des doublons
df = df.drop_duplicates()

# ### Gestion des valeurs manquantes
# Imputation des valeurs manquantes selon leur importance
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Self_Employed"] = df["Self_Employed"].fillna("No")
df["Credit_History"] = df["Credit_History"].fillna(0)

# Gestion des valeurs manquantes pour la cible
df_for_test = df[df["Loan_Status"].isnull()]
df = df.dropna(subset=["Loan_Status"])

# ### Gestion des encodages
encoders_config = {}

for col in df.columns:
    if col in ["Gender", "Married", "Self_Employed", "Loan_Status", "Dependents"]:
        # LabelEncoder pour les colonnes binaires ou multiclasses simples
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders_config[col] = {"type": "LabelEncoder", "encoder": le}
    elif col == "Education":
        # Mapping manuel
        mapping = {"Graduate": 0, "Not Graduate": 1}
        df[col] = df[col].map(mapping)
        encoders_config[col] = {"type": "map", "mapping": mapping}
    elif col == "Property_Area":
        # One-Hot Encoding
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1).drop(columns=[col])
        encoders_config[col] = {"type": "get_dummies", "columns": dummies.columns.tolist()}

with open(ENCODER_FILE, "wb") as file:
    pickle.dump(encoders_config, file)

# Sauvegarder les datasets intermédiaires
df.to_csv(WITHOUT_MISSING_FILE, index=False)
df_for_test.to_csv(FOR_TEST_FILE, index=False)

# ### Gestion des valeurs aberrantes
numeric_columns = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

# Suppression ou transformation des valeurs aberrantes
df_clean = df.copy()

# ApplicantIncome : Capping et transformation logarithmique
Q1 = df_clean["ApplicantIncome"].quantile(0.25)
Q3 = df_clean["ApplicantIncome"].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df_clean["ApplicantIncome"] = df_clean["ApplicantIncome"].clip(upper=upper_bound)
df_clean["ApplicantIncome_Log"] = np.log1p(df_clean["ApplicantIncome"])

# CoapplicantIncome : Capping et transformation logarithmique
Q1 = df_clean["CoapplicantIncome"].quantile(0.25)
Q3 = df_clean["CoapplicantIncome"].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df_clean["CoapplicantIncome"] = df_clean["CoapplicantIncome"].clip(upper=upper_bound)
df_clean["CoapplicantIncome_Log"] = np.log1p(df_clean["CoapplicantIncome"])

# LoanAmount : Suppression des valeurs < 25, Capping et transformation logarithmique
Q1 = df_clean["LoanAmount"].quantile(0.25)
Q3 = df_clean["LoanAmount"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df_clean[df_clean["LoanAmount"] >= lower_bound]
df_clean["LoanAmount"] = df_clean["LoanAmount"].clip(upper=upper_bound)
df_clean["LoanAmount_Log"] = np.log1p(df_clean["LoanAmount"])

# Loan_Amount_Term : Filtrage des valeurs non égales à 360
df_clean = df_clean[df_clean["Loan_Amount_Term"] == 360.0]

# Dernier filtre pour LoanAmount
df_clean = df_clean[df_clean["LoanAmount"] >= 25]

# Sauvegarder le dataset final pour l'entraînement
df_clean.to_csv(CLEAN_FOR_TRAINING, index=False)