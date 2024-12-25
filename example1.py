import numpy as np
import pandas as pd
from openai import AzureOpenAI
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load API key from local file
with open("key", "r") as file:
    api_key = file.read().strip()

# Set up Azure OpenAI service
api_version = "2024-02-01"
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint="https://openaiwsdev-gpt4-2.openai.azure.com/",
)

# Load the California Housing dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name="MedHouseVal")

# Basic EDA
# Check for missing values
# print("Missing values in each column:\n", X.isnull().sum())

# Preprocessing
# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Performance evaluation
y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))


# Integration with GPT-4 on Azure
def get_gpt4_decision(metrics):
    system_message = "Given the model performance metrics, suggest the best hyperparameters for n_estimators and max_depth for a Random Forest Regressor."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": str(metrics)},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
        max_tokens=10,
    )
    suggestion = response.choices[0].message.content.strip().lower()
    print(f"GPT-4 Suggestion: {suggestion}")
    return suggestion


metrics = {
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "MAE": mean_absolute_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred),
}

suggestion = get_gpt4_decision(metrics)

# Hyperparameter tuning based on GPT-4's suggestion
try:
    params = eval(suggestion)
    model.set_params(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("After tuning - RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("After tuning - MAE:", mean_absolute_error(y_test, y_pred))
    print("After tuning - R²:", r2_score(y_test, y_pred))
except Exception as e:
    print(f"Error in applying GPT-4's suggestion: {e}")
