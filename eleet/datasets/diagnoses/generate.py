from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from eleet.database import Database, Table, TextCollection, TextCollectionLabels


GENERATE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("user", """
Given the template '<name> was diagnosed with <health diagnosis> and his/her computer was diagnosed with <computer problem>',
generate {num_samples} short sentences that follow the template. For instance, 'Alice was diagnosed with diabetes and
her computer was diagnosed with a trojan horse'.
Be creative and generate diverse examples with different names, health diagnoses and computer problems.
Only output the generated examples without any explanations in an ordered list:
1. sentence 1.
2. sentence 2.
...
{num_samples}. sentence {num_samples}.
""".strip())])


def generate_data(
        db_dir=Path("datasets/diagnoses"),
        num_iterations_train=20,
        num_iterations_test=5,
        num_samples=50):

    output_path_train = db_dir / "generated_train.txt"
    output_path_test = db_dir / "generated_test.txt"
    with output_path_train.open("w") as file:
        for data_split in generate_data_split(num_iterations_train, num_samples):
            file.write(data_split)
    with output_path_test.open("w") as file:
        for data_split in generate_data_split(num_iterations_test, num_samples):
            file.write(data_split)


def generate_data_split(num_iterations, num_samples):
    model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.5)  # type: ignore
    generate_chain = GENERATE_TEMPLATE | model
    for _ in range(num_iterations):
        result_msg = generate_chain.invoke({"num_samples": num_samples})
        result_content = result_msg.content
        print("Result:", result_content)
        yield result_content


def load(db_dir, split):
    input_path = db_dir / f"generated_{split}.txt"
    with input_path.open("r") as file:
        data, alignments, texts = parse_result(file.read())
    return data, alignments, texts


def parse_result(result: str):
    regexp = re.compile(r"(\d+)\. (.*) was diagnosed with (.*) and (his|her) computer was diagnosed with (.*).")
    collected_data = []
    collected_texts = []
    collected_alignments = []

    for line in result.split("\n"):
        match = regexp.match(line)
        if not match:
            print(f"Skipping line: {line}")
            continue
        name = match.group(2)
        health_diagnosis = match.group(3)
        computer_problem = match.group(5)
        offset = match.start(2)
        name_alignment = tuple([x - offset for x in match.span(2)])
        health_diagnosis_alignment = tuple([x - offset for x in match.span(3)])
        computer_problem_alignment = tuple([x - offset for x in match.span(5)])
        collected_data.append((name, health_diagnosis, computer_problem))
        collected_texts.append(line[match.start(2):match.end(5) + 1])
        collected_alignments.append((name_alignment, health_diagnosis_alignment, computer_problem_alignment))
    collected_data = pd.DataFrame(collected_data, columns=["name", "health_diagnosis", "computer_problem"])
    collected_alignments = pd.DataFrame(collected_alignments, columns=["name", "health_diagnosis", "computer_problem"])
    return collected_data, collected_alignments, collected_texts


def load_diagnoses(db_dir, split, num_sentences_per_report=2):
    data, alignments, texts = load(db_dir, split)
    train_data, _, train_texts = load(db_dir, "train")

    health_union_evidence = train_data[["name", "health_diagnosis"]].copy()  # type: ignore
    health_union_evidence.rename({"health_diagnosis": "diagnosis"}, inplace=True, axis="columns")
    health_union_evidence["report_number"] = np.arange(len(train_texts)) - len(train_texts)

    computer_union_evidence = train_data[["name", "computer_problem"]].copy()  # type: ignore
    computer_union_evidence.rename({"computer_problem": "diagnosis"}, inplace=True, axis="columns")
    computer_union_evidence["report_number"] = np.arange(len(train_texts)) - len(train_texts)

    normed_health = data[["name", "health_diagnosis"]].copy()  # type: ignore
    normed_health.rename({"health_diagnosis": "diagnosis"}, inplace=True, axis="columns")
    alignments_health = alignments[["name", "health_diagnosis"]].copy()  # type: ignore
    alignments_health.rename({"health_diagnosis": "diagnosis"}, inplace=True, axis="columns")

    normed_computer = data[["name", "computer_problem"]].copy()  # type: ignore
    normed_computer.rename({"computer_problem": "diagnosis"}, inplace=True, axis="columns")
    alignments_computer = alignments[["name", "computer_problem"]].copy()  # type: ignore
    alignments_computer.rename({"computer_problem": "diagnosis"}, inplace=True, axis="columns")

    texts = pd.DataFrame([(i // num_sentences_per_report, x) for i, x in enumerate(texts)],
                         columns=["report_number", "text"])
    normed_health["report_number"] = normed_computer["report_number"] = texts["report_number"]  # type: ignore
    normed_health.set_index(["report_number", "name"], inplace=True, drop=False)
    normed_health.drop("report_number", axis=1, inplace=True)
    normed_computer.set_index(["report_number", "name"], inplace=True, drop=False)
    normed_computer.drop("report_number", axis=1, inplace=True)
    alignments_health.index = normed_health.index
    alignments_computer.index = normed_computer.index

    offset = 0
    for i in range(len(texts)):
        k = i % num_sentences_per_report
        if k == 0:
            offset = 0

        alignments_health.iloc[i] = alignments_health.iloc[i].map(lambda x: tuple([y + offset for y in x]))
        alignments_computer.iloc[i] = alignments_computer.iloc[i].map(lambda x: tuple([y + offset for y in x]))
        offset += len(texts.iloc[i]["text"]) + 1

    texts = texts.groupby("report_number").agg(lambda x: " ".join(x)).reset_index(drop=False)  # type: ignore
    normed_health = normed_health.applymap(lambda x: [x])
    normed_computer = normed_computer.applymap(lambda x: [x])
    alignments_health = alignments_health.applymap(lambda x: [tuple(x)])
    alignments_computer = alignments_computer.applymap(lambda x: [tuple(x)])

    tables = [
        Table(name="computer_problems", data=computer_union_evidence, key_columns=["report_number"]),  # type: ignore
        Table(name="health_issues", data=health_union_evidence, key_columns=["report_number"]),  # type: ignore
    ]
    labels_computer = TextCollectionLabels(normed=normed_computer, alignments=alignments_computer)  # type: ignore
    labels_health = TextCollectionLabels(normed=normed_health, alignments=alignments_health)  # type: ignore
    reports = TextCollection(name="reports", data=texts, key_columns=["report_number"])  # type: ignore
    reports.setup_text_table("health_issues_new", attributes=["name", "diagnosis"], multi_row=True,
                             identifying_attribute="name", labels=labels_health,
                             force_single_value_attributes={"name", "diagnosis"},
                             simulate_single_table=True)
    reports.setup_text_table("computer_problems_new", attributes=["name", "diagnosis"], multi_row=True,
                             identifying_attribute="name", labels=labels_computer,
                             force_single_value_attributes={"name", "diagnosis"},
                             simulate_single_table=True)
    db = Database(
        name="diagnoses", tables=tables, texts = [reports]
    )
    return db


def test_alignments(reports: TextCollection):
    for text_table in reports.text_tables.values():
        assert text_table.labels is not None
        normed = text_table.labels.normed
        alignments = text_table.labels.alignments
        texts = reports.data

        for i in texts.index:
            for c in normed.columns:
                for n, a in zip(normed.loc[[i], c], alignments.loc[[i], c]):
                    t = texts.loc[i, "text"]
                    assert len(n) == len(a) == 1
                    assert n[0] == t[a[0][0]:a[0][1]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--db-dir", type=Path, default=Path("datasets/diagnoses"))
    parser.add_argument("--num-iterations-train", type=int, default=20)
    parser.add_argument("--num-iterations-test", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()

    if args.generate_data:
        generate_data(
            db_dir=args.db_dir,
            num_iterations_train=args.num_iterations_train,
            num_iterations_test=args.num_iterations_test,
            num_samples=args.num_samples)

    db = load_diagnoses(args.db_dir, "test")
    test_alignments(db.texts["reports"])


