import json
import os
from typing import List

import pandas as pd
from openai import AzureOpenAI
from pydantic import BaseModel
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_gpt4_decision(metrics):
    system_message = (
        "Given the model performance metrics and the training log, suggest the best "
        "hyperparameters for n_estimators and max_depth for a Random Forest Regressor. "
        "Only suggest new hyperparameters that have not been tested before. If no new "
        "hyperparameters can be suggested, provide a decision to stop optimization with a simple yes or no."
    )

    log = load_log()

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message,
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": metrics}]},
        {"role": "user", "content": [{"type": "text", "text": f"Training log: {log}"}]},
    ]

    class Param(BaseModel):
        n_estimators: float
        max_depth: float

    class Schema(BaseModel):
        decision: str
        params: List[Param]

    # Generate the completion
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2",
        messages=messages,
        temperature=1,
        response_format=Schema,
    )

    suggestion = completion.choices[0].message.content.strip().lower()
    suggestion = json.loads(suggestion)
    return suggestion


LOG_FILE = "training_log.json"


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as file:
            return json.load(file)
    return []


def save_log(log):
    with open(LOG_FILE, "w") as file:
        json.dump(log, file, indent=2)


def log_exists(log, params):
    return any(entry["params"] == params for entry in log)


with open("key", "r") as file:
    api_key = file.read().strip()

# Set up Azure OpenAI service
api_version = "2024-08-01-preview"
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint="https://openaiwsdev-gpt4-2.openai.azure.com/",
)

# Load the California Housing dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name="MedHouseVal")

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Performance evaluation
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

metrics = f"mae {mean_absolute_error(y_test, y_pred)} r2 {r2_score(y_test, y_pred)}"
print("Metrics:", metrics)

log = load_log()
current_params = {
    "n_estimators": model.get_params()["n_estimators"],
    "max_depth": model.get_params()["max_depth"],
}

if not log_exists(log, current_params):
    log.append({"params": current_params, "metrics": metrics})
    save_log(log)

metrics = f"mae {mean_absolute_error(y_test, y_pred)} r2 {r2_score(y_test, y_pred)}"
print("Metrics:", metrics)

tries = 0
max_tries = 15

while tries < max_tries:
    suggestion = get_gpt4_decision(metrics)

    if suggestion["decision"] == "no":
        print("No further optimization needed based on GPT-4's suggestion.")
        break

    print("Optimizing hyperparameters based on GPT-4's suggestion.")
    suggested_params = suggestion["params"][0]
    if not log_exists(log, suggested_params):
        # Validate hyperparameters
        if (
            isinstance(suggested_params["max_depth"], int)
            and suggested_params["max_depth"] > 0
        ):
            model.set_params(**suggested_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("After tuning - MAE:", mean_absolute_error(y_test, y_pred))
            print("After tuning - R²:", r2_score(y_test, y_pred))
            metrics = f"mae {mean_absolute_error(y_test, y_pred)} r2 {r2_score(y_test, y_pred)}"
            log.append({"params": suggested_params, "metrics": metrics})
            save_log(log)
        else:
            print(f"Invalid hyperparameters suggested: {suggested_params}")
    else:
        print("Suggested hyperparameters have already been tested.")

    tries += 1

if tries == max_tries:
    print("Reached maximum number of optimization attempts.")
