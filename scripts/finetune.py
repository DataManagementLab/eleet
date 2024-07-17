import os
from pathlib import Path
from eleet.datasets.aviation.aviation import load_aviation
from eleet.datasets.corona.corona import load_corona
from eleet.datasets.rotowire.rotowire import load_rotowire_legacy
from eleet.datasets.trex.trex import load_trex_legacy
from eleet.datasets.diagnoses.generate import load_diagnoses
from eleet.methods.openai.engine import LLM_COST
from eleet.methods.operator import Join, MMJoin, MMUnion


OPENAI_METHODS = tuple(LLM_COST.keys())

def get_model(model_name):
    if model_name == "eleet":
        from eleet.methods.eleet.finetuning import ELEETFinetuneEngine
        from eleet.methods.eleet.preprocessor import ELEETPreprocessor
        from transformers import BertTokenizerFast
        from eleet_pretrain.model.config import VerticalEleetConfig
        config = VerticalEleetConfig(max_num_cols=20)
        tokenizer = BertTokenizerFast.from_pretrained(config.base_model_name)
        preprocessor = ELEETPreprocessor(config=config, tokenizer=tokenizer)
        engine = ELEETFinetuneEngine(config=config, tokenizer=tokenizer, raise_exceptions=True,
                                    cache_dir=Path("predictions") / "finetuning" / "cache")
        return engine, preprocessor
    if model_name == "t2t":
        from eleet.methods.text_to_table.finetuning import T2TFinetuningEngine
        from eleet.methods.text_to_table.preprocessor import T2TPreprocessor
        preprocessor = T2TPreprocessor(encoder_json=Path("text_to_table") / "encoder.json",
                                       vocab_bpe=Path("text_to_table") / "vocab.bpe")
        engine = T2TFinetuningEngine("bart.base", cache_dir=Path("predictions") / "finetuning" / "cache")
        return engine, preprocessor

    if model_name in OPENAI_METHODS:
        from eleet.methods.llama.preprocessor import LLMPreprocessor
        from eleet.methods.openai.finetuning import OpenAIFinetuningEngine
        preprocessor = LLMPreprocessor(train_db=None, num_samples=0, finetune_split_size=0)
        engine = OpenAIFinetuningEngine(llm=model_name, cache_dir=Path("predictions") / "finetuning" / "cache")
        return engine, preprocessor

    if model_name == "llama":
        from eleet.methods.llama.preprocessor import LLMPreprocessor
        from eleet.methods.llama.finetuning import LLamaFinetuningEngine
        preprocessor = LLMPreprocessor(train_db=None, num_samples=0, finetune_split_size=0)
        engine = LLamaFinetuningEngine(
            ckpt_dir="/mnt/labstore/SIGs/ML/llama-2-7b-chat/",
            tokenizer_path="/home/murban/llama/tokenizer.model",
            cache_dir=Path("predictions") / "finetuning" / "cache"
        )
        return engine, preprocessor
    raise ValueError(f"Unknown model {model_name}")

def get_rotowire_queries():
    db_train = load_rotowire_legacy(Path(__file__).parents[1] / "datasets" / "rotowire", "train")
    db_valid = load_rotowire_legacy(Path(__file__).parents[1] / "datasets" / "rotowire", "valid")

    query_plan_player_join = MMJoin(operands=[
        Join(operands=["player_info", "player_to_reports"], join_key="name"),
        "reports.Player",
    ], join_key="Game ID")
    query_plan_player_union = MMUnion(operands=[
        "player_stats",
        "reports.Player",
    ])
    query_plan_team_join = MMJoin(operands=[
        Join(operands=["team_info", "team_to_reports"], join_key="name"),
        "reports.Team",
    ], join_key="Game ID")
    query_plan_team_union = MMUnion(operands=[
        "team_stats",
        "reports.Team",
    ])

    return db_train, db_valid, (query_plan_player_join, query_plan_player_union,
                                query_plan_team_join, query_plan_team_union)


def get_trex_queries():
    db_train = load_trex_legacy(Path(__file__).parents[1] / "datasets" / "trex", "train")
    db_valid = load_trex_legacy(Path(__file__).parents[1] / "datasets" / "trex", "valid")

    union_nobel_personal = MMUnion(operands=[
        "nobel-Personal-union",
        "nobel_reports.Personal",
    ])

    union_nobel_career = MMUnion(operands=[
        "nobel-Career-union",
        "nobel_reports.Career",
    ])

    union_countries_geography = MMUnion(operands=[
        "countries-Geography-union",
        "countries_reports.Geography",
    ])

    union_countries_politics = MMUnion(operands=[
        "countries-Politics-union",
        "countries_reports.Politics",
    ])

    union_skyscrapers_location = MMUnion(operands=[
        "skyscrapers-Location-union",
        "skyscrapers_reports.Location",
    ])

    union_skyscrapers_history = MMUnion(operands=[
        "skyscrapers-History-union",
        "skyscrapers_reports.History",
    ])

    #####

    join_nobel_personal = MMJoin(operands=[
        "nobel-Career-join",
        "nobel_reports.Personal",
    ], join_key="index")

    join_nobel_career = MMJoin(operands=[
        "nobel-Personal-join",
        "nobel_reports.Career",
    ], join_key="index")

    join_countries_geography = MMJoin(operands=[
        "countries-Politics-join",
        "countries_reports.Geography",
    ], join_key="index")

    join_countries_politics = MMJoin(operands=[
        "countries-Geography-join",
        "countries_reports.Politics",
    ], join_key="index")

    join_skyscrapers_location = MMJoin(operands=[
        "skyscrapers-History-join",
        "skyscrapers_reports.Location",
    ], join_key="index")

    join_skyscrapers_history = MMJoin(operands=[
        "skyscrapers-Location-join",
        "skyscrapers_reports.History",
    ], join_key="index")

    return db_train, db_valid, (union_nobel_personal,
                                union_nobel_career,
                                union_countries_geography,
                                union_countries_politics,
                                union_skyscrapers_history,
                                union_skyscrapers_location,
                                join_nobel_personal,
                                join_nobel_career,
                                join_countries_geography,
                                join_countries_politics,
                                join_skyscrapers_history,
                                join_skyscrapers_location)


def get_corona_queries():
    db_train = load_corona(Path(__file__).parents[1] / "datasets" / "corona", "train")
    db_valid = load_corona(Path(__file__).parents[1] / "datasets" / "corona", "test")
    union = MMUnion(operands=["corona_stats", "reports.summary"])
    return db_train, db_valid, (union,)


def get_aviation_queries():
    db_train = load_aviation(Path(__file__).parents[1] / "datasets" / "aviation", "train")
    db_valid = load_aviation(Path(__file__).parents[1] / "datasets" / "aviation", "test")
    union = MMUnion(operands=["incidents", "reports.incident"])
    join = MMJoin(operands=[
        Join(operands=["aircraft", "aircraft_to_reports"], join_key="aircraft_registration_number"),
        "reports.incident",
    ], join_key="report_number")

    return db_train, db_valid, (union, join)


def get_diagnoses_queries():
    db_train = load_diagnoses(Path(__file__).parents[1] / "datasets" / "diagnoses", "train")
    db_valid = load_diagnoses(Path(__file__).parents[1] / "datasets" / "diagnoses", "test")
    union_health = MMUnion(operands=["health_issues", "reports.health_issues_new"])
    union_computer = MMUnion(operands=["computer_problems", "reports.computer_problems_new"])
    return db_train, db_valid, (union_health, union_computer)


def main():
    dataset = os.environ["FINETUNE_DATASET"]
    model_name = os.environ["FINETUNE_MODEL"]
    func = globals()[f"get_{dataset}_queries"]
    db_train, db_valid, queries = func()
    engine, preprocessor = get_model(model_name)
    db_train.finetune(query_plans=queries, preprocessor=preprocessor, engine=engine,
                      db_valid=db_valid)


if __name__ == '__main__':
    main()
