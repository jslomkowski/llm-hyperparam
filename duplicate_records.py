import json
import re

import pandas as pd
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from openai import AzureOpenAI

from mdm_dr.utils import DATA, ROOT

conf_path = str(ROOT / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
api_key = conf_loader["credentials"]["api_key"]
api_version = "2024-02-01"


def fix_json_string(json_string):
    "foo"
    json_string = re.sub(r",(\s*[}\]])", r"\1", json_string)
    json_string = re.sub(r"\'", r'"', json_string)
    json_string = re.sub(r"(?<=\d),(?=\d)", r"", json_string)
    return json_string


def load_fixed_json(json_string):
    "foo"
    fixed_json_string = fix_json_string(json_string)
    try:
        return json.loads(fixed_json_string)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        return None


df = pd.read_parquet(DATA / "DD_Department_SOT_keep_first.parquet")
df_result = pd.DataFrame()
for i in range(0, len(df), 10):
    print(i)
    # i = 0
    df_c = df.iloc[i : i + 10]
    departments = df_c.to_json(orient="records")
    departments = json.dumps(departments)

    system_message = "You ar a data validation tool. Your job is to pick up any abnormal departament names from the list porvided. Make sure to pick up any plurar and non plural words, plurar and singular topology. verbs that have been tensed, missing letters or aditional letters or aditional spaces. Output should be a json dictionary with same input keys and values but aditional value should be added called verdict, telling what is wrong with the departament. If everything is ok with departament then do not output it"

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint="https://openaiwsdev-gpt4-2.openai.azure.com/",
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": []},
    ]

    messages[1]["content"].append({"type": "text", "text": departments})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
        max_tokens=1500,
        # response_format={"type": "json_object"},
    )

    data = response.choices[0].message.content
    json_match = re.search(r"```json\n(.*)\n```", data, re.DOTALL)
    json_part = json_match.group(1)
    try:
        json_response = json.loads(json_part)
    except:
        json_response = load_fixed_json(json_part)

    # Convert the dictionary to a pandas DataFrame
    df_result_part = pd.DataFrame(json_response)

    df_result_part = pd.merge(
        df_c, df_result_part[["ID", "vardict"]], on="ID", how="left"
    )
    df_result = pd.concat([df_result, df_result_part])

df_result.to_excel(DATA / "df_result.xlsx", index=False)
