import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI


NUM_REPORTS = 3
DATASETS = ["rotowire", "trex", "aviation", "corona"]

TEMPLATE_IDENTIFYING_ATTRIBUTE = ChatPromptTemplate.from_messages([
    ("user", """
I want to transform the information stored in texts into a table. The table should have the following columns: {columns}.
In a first step, please specify one of the columns that act as a document-level key, meaning that all extracted values should be
unique per document. Only output the name of that column without any explanations.

Sample of texts:
{texts}
""".strip())])

TEMPLATE_MULTI_ROW = ChatPromptTemplate.from_messages([
    ("user", """
I want to transform the information stored in texts into a table. The table should have the following columns: {columns}.
In a first step, please determine if one of many rows are extracted per text document. Please only output "many" or "one"
without any explanations.

Sample of texts:
{texts}
""".strip())])

COLUMNS = {
    "rotowire": {
        "player": [
            "Player Name", "Assists", "Points", "Total rebounds", "Steals", "Defensive rebounds",
            "Field goals attempted", "Field goals made", "Free throws attempted", "Free throws made",
            "Minutes played", "Personal fouls", "Turnovers", "Blocks", "Offensive rebounds", "Field goal percentage",
            "Free throw percentage"

        ],
        "team": [
            "Team Name", "Losses", "Total points", "Points in 4th quarter", "Wins", "Percentage of field goals",
            "Rebounds", "Number of team assists", "Points in 3rd quarter", "Turnovers", "Percentage of 3 points",
            "Points in 1st quarter", "Points in 2nd quarter"]
    },
    "trex": {
        "countries-Geography": [
            "name", "continent", "capital", "shares border with", "part of", "lowest point", "highest point",
            "located in or next to body of water"],
        "countries-Politics": [
            "name", "head of state", "head of government", "basic form of government", "inception", "member of",
            "diplomatic relation", "contains administrative territorial entity"],
        "nobel-Career": [
            "name", "field of work", "occupation", "employer", "award received", "nominated for", "member of",
            "educated at"],
        "nobel-Personal": [
            "name", "place of birth", "date of birth", "place of death", "date of death", "country of citizenship",
            "native language", "sex or gender"],
        "skyscrapers-History": [
            "name", "inception", "date of official opening", "architectural style", "use", "heritage designation",
            "occupant", "significant event"],
        "skyscrapers-Location": [
            "name", "country", "coordinate location", "located in the administrative territorial entity", "architect",
            "main building contractor", "located on street", "location"]
    },
    "corona": {
        "corona": [
             "date",  # date of the report
             "new_cases",  # number of new cases
             "new_deaths",  # number of new deaths
             "incidence",  # 7-day incidence
             "patients_intensive_care",  # number of people in intensive care
             "vaccinated",  # number of people that have been vaccinated at least once
             "twice_vaccinated"  # number of people that have been vaccinated twice
             ],
    },
    "aviation": {
        "aviation": [
            "event_date",  # date of the event
            "location_city",  # city or place closest to the site of the event
            "location_state",  # state the city is located in
            "airport_code",  # code of the airport
            "airport_name",  # airport name
            "aircraft_damage",  # severity of the damage to the aircraft
            "weather_condition"  # weather conditions at the time of the event
        ],
    }
}



def get_texts_from_db(path, table_name):
    complete_path = f"{path}/reports_table.json"
    if not os.path.exists(complete_path):
        complete_path = f"{path}/{table_name}-reports.json"
    if not os.path.exists(complete_path):
        raise Exception(f"File {complete_path} does not exist")
    df = pd.read_json(complete_path)
    reports = df.iloc[:NUM_REPORTS][df.columns[-1]].tolist() # type: ignore
    return reports


def get_texts_from_raw_documents(path):
    reports = []
    files = os.listdir(path)[:NUM_REPORTS]
    for file in files:
        with open(f"{path}/{file}", "r") as f:
            reports.append(f.read())
    return reports


def get_texts():
    texts = {}
    for dataset in DATASETS:
        texts[dataset] = {}
        for table_name in COLUMNS[dataset].keys():
            dirs = os.listdir(f"datasets/{dataset}")
            if "db" in dirs:
                texts[dataset][table_name] = get_texts_from_db(f"datasets/{dataset}/db/test", table_name)
            elif "raw-documents" in dirs:
                texts[dataset][table_name] = get_texts_from_raw_documents(f"datasets/{dataset}/raw-documents")
            else:
                raise Exception(f"Dataset {dataset} does not contain a db or raw-documents directory")
    return texts


if __name__ == "__main__":
    model = ChatOpenAI(model="gpt-3.5-turbo-0125")
    texts = get_texts()
    for dataset in texts.keys():
        for table_name, text in texts[dataset].items():
            columns = COLUMNS[dataset][table_name]

            chain_identifying_attribute = TEMPLATE_IDENTIFYING_ATTRIBUTE | model
            chain_multi_row = TEMPLATE_MULTI_ROW | model

            result_identifying_attribute = chain_identifying_attribute.invoke({"columns": columns, "texts": text})
            result_multi_row = chain_multi_row.invoke({"columns": columns, "texts": text})

            print(f"Dataset: {dataset}, Table: {table_name}")
            print("Identifying attribute:", result_identifying_attribute.content)
            print("Multi row:", result_multi_row.content)

