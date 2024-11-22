import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.metrics import make_scorer, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import pickle

DATA_PATH = "data/"
CLEAN_FOR_TRAINING = DATA_PATH + "processed/data_clean_for_training.csv"
PICKLE_PATH = "outputs/pickle/"
TRAINING_FILE = PICKLE_PATH + "training_data.pkl"

# Load data
df = pd.read_csv(filepath_or_buffer=CLEAN_FOR_TRAINING, sep=",", header=0)  # Replace with your dataset

# Splitting features and target
log_columns = [col for col in df.columns if "_Log" in col]
original_columns = [col for col in df.columns if "_Log" not in col and f"{col}_Log" not in log_columns]
# Supprimer 'Loan_ID' de la liste des colonnes originales
columns_to_use = [col for col in (original_columns + log_columns) if col != "Loan_ID"]
X = df[columns_to_use]
y = df["Loan_Status"]

# Division train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définir les hyperparamètres à tester
param_grid = {
    'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

}
loo = LeaveOneOut()
# Configurer GridSearchCV
grid_search = GridSearchCV(GaussianNB(), param_grid, cv=loo, scoring=make_scorer(precision_score, zero_division=1))
# Lancer GridSearchCV
grid_search.fit(X_train_scaled, y_train)

gaussianNB = GaussianNB(**grid_search.best_params_)
gaussianNB.fit(X_train_scaled, y_train)

gaussianNB = GaussianNB(**grid_search.best_params_)
gaussianNB.fit(X_train_scaled, y_train)


# Sauvegarder le scaler et l'ordre des colonnes dans un fichier pickle
with open(TRAINING_FILE, "wb") as f:
    pickle.dump({"model": gaussianNB, "scaler": scaler, "columns": columns_to_use}, f)