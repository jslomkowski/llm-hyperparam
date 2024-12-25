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
        "Given the model performance metrics and the training log, suggest the"
        "best hyperparameters for a Random Forest Regressor. The optimization"
        "goal is to maximize model performance and reduce overfitting. Only"
        "suggest new hyperparameters that have not been tested before. If no"
        "new hyperparameters can be suggested, provide a decision to stop or"
        "proceed with optimization with a simple yes or no."
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
        {"role": "user", "content": [{"type": "text", "text": f"Training log: {log}"}]},
    ]

    class Param(BaseModel):
        param_name: str
        param_value: float

    class Schema(BaseModel):
        decision: str
        params: List[Param]
        reason: str

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


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as file:
            content = file.read().strip()
            if content:
                return json.loads(content)
    return []


def save_log(log):
    with open(LOG_FILE, "w") as file:
        json.dump(log, file, indent=2)


def log_exists(log, params):
    return any(entry["params"] == params for entry in log)


def log_metrics(log, params, metrics, reason, error=""):
    log_entry = {
        "params": params,
        "metrics": metrics,
        "reason": reason,
        "error": error,
    }
    log.append(log_entry)
    save_log(log)


def calculate_metrics(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    return train_mae, train_r2, test_mae, test_r2


LOG_FILE = "training_log.json"

with open("key", "r") as file:
    api_key = file.read().strip()

# Set up Azure OpenAI service
api_version = "2024-08-01-preview"
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint="https://openaiwsdev-gpt4-2.openai.azure.com/",
)


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


train_mae, train_r2, test_mae, test_r2 = calculate_metrics(
    model, X_train, y_train, X_test, y_test
)
metrics = (
    f"train_mae {train_mae} train_r2 {train_r2} test_mae {test_mae} test_r2 {test_r2}"
)
print("Metrics:", metrics)

log = load_log()
current_params = model.get_params()

if not log_exists(log, current_params):
    log.append(
        {"params": current_params, "metrics": metrics, "reason": "Initial model"}
    )
    save_log(log)

tries = 0
max_tries = 15

while tries < max_tries:
    suggestion = get_gpt4_decision(metrics)

    if suggestion["decision"] == "no":
        print(
            "No further optimization needed based on GPT-4's suggestion.",
            suggestion["reason"],
        )
        log.append({"params": [], "metrics": [], "reason": suggestion["reason"]})
        save_log(log)
        break

    print("Optimizing hyperparameters based on GPT-4's suggestion.")
    suggested_params = {
        param["param_name"]: param["param_value"] for param in suggestion["params"]
    }
    if not log_exists(log, suggested_params):
        # Validate hyperparameters
        try:
            model = RandomForestRegressor(random_state=42)
            model.set_params(**suggested_params)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            metrics = (
                f"train_mae {train_mae} train_r2 {train_r2} "
                f"test_mae {test_mae} test_r2 {test_r2}"
            )
            log_metrics(log, suggested_params, metrics, suggestion["reason"])
        except ValueError as e:
            error_message = str(e)
            print(
                f"Invalid hyperparameters suggested: {suggested_params}. Error: {error_message}"
            )
            log_metrics(
                log, suggested_params, metrics, suggestion["reason"], error_message
            )
    else:
        print("Suggested hyperparameters have already been tested.")

    tries += 1

if tries == max_tries:
    print("Reached maximum number of optimization attempts.")
