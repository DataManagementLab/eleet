from eleet.datasets.rotowire.pairwise_model import PairwiseModel
from eleet.datasets.rotowire.match import Match
from eleet.datasets.rotowire.rotowire_dataset import Rotowire
from sklearn.ensemble import RandomForestClassifier
import joblib
import fuzzysearch
import re
import logging
import itertools
import numpy as np

logger = logging.getLogger(__name__)
model = PairwiseModel(RandomForestClassifier())

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

NUMBER_WORDS = {
    "0": ("zero",),
    "1": ("one", "a", "an", "once"),
    "2": ("two", "twice", "pair"),
    "3": ("three",),
    "4": ("four",),
    "5": ("five",),
    "6": ("six",),
    "7": ("seven",),
    "8": ("eight",),
    "9": ("nine",),
    "10": ("ten",),
    "11": ("eleven",),
    "12": ("twelve",),
    "20": ("twenty",),
    "30": ("thirty",),
    "40": ("forty",),
    "50": ("fifty",),
    "60": ("sixty",),
    "70": ("seventy",),
    "80": ("eighty",),
    "90": ("ninety",),
    "100": ("hundred",),
}

ALL_NUMBER_WORDS = set(NUMBER_WORDS) | {y for x in NUMBER_WORDS.values() for y in x}

COL_ALIAS = {
    'Minutes played': ('minutes',),
    'Assists': ('assists', 'assist'),
    'Points': ('points', 'point'),
    'Total rebounds': ('rebounds', 'rebound', 'rebounding'),
    'Steals': ('steal', 'stole', 'stolen', 'stealing', 'steals'),
    'Blocks': ('block', 'blocked', 'blocks', 'blocking'),
    'Defensive rebounds': ('defensive', 'rebounds', 'rebound', 'rebounding'),
    'Personal fouls': ('fouls', 'foul'),
    'Turnovers': ('coughing up the ball', 'loosing the ball', 'turnovers', 'turnover'),
    'Offensive rebounds': ('offensive', 'rebounds', 'rebound', 'rebounding'),
    '3-pointers made': ('threes', '3Pt', 'arc', '3 - pointers', 'three - pointers', '3 - pointer', 'three - pointer', 'from deep', 'three - point', '3 - point'),
    '3-pointers attempted': ('threes', '3Pt', 'arc', '3 - pointers', 'three - pointers', '3 - pointer', 'three - pointer', 'from deep', 'three - point', '3 - point'),
    '3-pointer percentage': ('percent', 'percentage', 'threes', '3Pt', 'arc', '3 - pointers', 'three - pointers', '3 - pointer', 'three - pointer', 'from deep', 'three - point', '3 - point'),
    'Field goals made': ('shot', 'shooting', 'from the field', 'shots', 'FG', 'field goals', 'field goal', 'boards', 'board', 'from the floor'),
    'Field goals attempted': ('shot', 'shooting', 'from the field', 'shots', 'FG', 'field goals', 'field goal', 'boards', 'board', 'from the floor'),
    'Field goal percentage': ('percent', 'percentage', 'shot', 'shooting', 'from the field', 'shots', 'FG', 'field goals', 'field goal', 'boards', 'board', 'from the floor'),
    'Free throws made': ('free - throw', 'free - throws', 'FT', 'free throw', 'free throws', 'the line', 'foul line'),
    'Free throws attempted': ('free - throw', 'free - throws', 'FT', 'free throw', 'free throws', 'the line', 'foul line'),
    'Free throw percentage': ('percent', 'percentage', 'free - throw', 'free - throws', 'FT', 'free throw', 'free throws', 'the line', 'foul line'),
    'Number of team assists': ('assists', 'team assists'),
    'Losses': ('lost', 'loose'),
    'Wins': ('won', 'win'),
    'Total points': ('points', 'total', 'score'),
    'Points in 1st quarter': ('1st', 'first', 'quarter', 'section', 'period', '12 minutes', 'twelve minutes', 'frame'),
    'Points in 2nd quarter': ('2nd', 'second', 'quarter', 'section', 'period', '12 minutes', 'twelve minutes', 'frame'),
    'Points in 3rd quarter': ('3rd', 'third', 'quarter', 'section', 'period', '12 minutes', 'twelve minutes', 'frame'),
    'Points in 4th quarter': ('4th', 'fourth', 'quarter', 'section', 'period', 'last', 'final', '12 minutes', 'twelve minutes', 'frame'),
    'Rebounds': ('rebounds', 'rebound', 'rebounding'),
    'Percentage of 3 points': ('percent', 'percentage', 'threes', '3Pt', 'arc', '3 - pointers', 'three - pointers', '3 - pointer', 'three - pointer', 'from deep', 'three - point', '3 - point'),
    'Percentage of field goals': ('percent', 'percentage', 'shot', 'shooting', 'from the field', 'shots', 'FG', 'field goals', 'field goal', 'boards', 'board', 'from the floor'),
    'Percentage of free throws': ('percent', 'percentage', 'free - throw', 'free - throws', 'FT', 'free throw', 'free throws', 'the line', 'foul line')

}

COLS = sorted(COL_ALIAS.keys())

COL_FEATURES = {
    'Minutes played':            (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Assists':                   (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Points':                    (1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Total rebounds':            (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Steals':                    (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Blocks':                    (1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Defensive rebounds':        (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Personal fouls':            (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Turnovers':                 (1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Offensive rebounds':        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    '3-pointers made':           (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    '3-pointers attempted':      (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    '3-pointer percentage':      (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Field goals made':          (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Field goals attempted':     (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Field goal percentage':     (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Free throws made':          (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Free throws attempted':     (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Free throw percentage':     (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Number of team assists':    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Losses':                    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Wins':                      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    'Total points':              (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1),
    'Points in 1st quarter':     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1),
    'Points in 2nd quarter':     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1),
    'Points in 3rd quarter':     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
    'Points in 4th quarter':     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1),
    'Rebounds':                  (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1),
    'Percentage of 3 points':    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1),
    'Percentage of field goals': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
    'Percentage of free throws': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
}

TEAM_ALIASES = {
    "Celtics": ["Boston Celtics", "Boston"],
    "Nets": ["Brooklyn Nets", "Brooklyn"],
    "Knicks": ["New York Knicks", "New York"],
    "76ers": ["Philadelphia 76ers", "Philadelphia", "Sixers"],
    "Raptors": ["Toronto Raptors", "Toronto"],
    "Nuggets": ["Denver Nuggets", "Denver"],
    "Timberwolves": ["Minnesota Timberwolves", "Minnesota", "Wolves", "T-Wolves"],
    "Thunder": ["Oklahoma City Thunder", "Oklahoma City"],
    "Trail Blazers": ["Portland Trail Blazers", "Portland"],
    "Jazz": ["Utah Jazz", "Utah"],
    "Bulls": ["Chicago Bulls", "Chicago"],
    "Cavaliers": ["Cleveland Cavaliers", "Cleveland"],
    "Pistons": ["Detroit Pistons", "Detroit"],
    "Pacers": ["Indiana Pacers", "Indiana"],
    "Bucks": ["Milwaukee Bucks", "Milwaukee"],
    "Warriors": ["Golden State Warriors", "Golden State", "San Francisco"],
    "Clippers": ["LA Clippers", "LA", "Los Angeles"],
    "Lakers": ["Los Angeles Lakers", "LA", "Los Angeles"],
    "Suns": ["Phoenix Suns", "Phoenix"],
    "Kings": ["Sacramento Kings", "Sacramento"],
    "Hawks": ["Atlanta Hawks", "Atlanta"],
    "Hornets": ["Charlotte Hornets", "Charlotte"],
    "Heat": ["Miami Heat", "Miami"],
    "Magic": ["Orlando Magic", "Orlando"],
    "Wizards": ["Washington Wizards", "Washington"],
    "Mavericks": ["Dallas Mavericks", "Dallas"],
    "Rockets": ["Houston Rockets", "Houston"],
    "Grizzlies": ["Memphis Grizzlies", "Memphis"],
    "Pelicans": ["New Orleans Pelicans", "New Orleans"],
    "Spurs": ["San Antonio Spurs", "San Antonio"]
}

remove_suffixes = ["Jr.", "Sr.", "I", "II", "III", "IV", "V", "VI", "VII", "IIX", "IX", "X"]


class Aligner():
    def __init__(self, data_dir, cache_dir, model_path=None, split="train"):
        self.cache_dir = cache_dir
        self.dataset = Rotowire(split, data_dir)
        self.split = split
        self.model = joblib.load(model_path) if model_path is not None else None

    def align(self, player_df, team_df, line_text):
        assert self.model, "Model must be provided in __init__"
        player_df_name_matches, player_df_col_matches, player_df_value_matches = self.get_matches(player_df, line_text)
        team_df_name_matches, team_df_col_matches, team_df_value_matches = self.get_matches(team_df, line_text)
        result_player = self._align(player_df, line_text, player_df_name_matches,
                                    player_df_col_matches, player_df_value_matches, team_df_name_matches)
        result_team = self._align(team_df, line_text, team_df_name_matches,
                                  team_df_col_matches, team_df_value_matches, team_df_name_matches)
        return result_player, result_team

    def _align(self, df, line_text, df_name_matches, df_col_matches, df_value_matches, team_df_name_matches):
        result = df.copy()

        for i, row in df.iterrows():
            for col in df.columns[1:]:
                value = row[col]
                if not value:
                    continue

                value_features = self.compute_features(line_text, value, row["Name"], col,
                                                       df_value_matches, df_name_matches,
                                                       df_col_matches, team_df_name_matches)
                pred = self.model.predict(value_features)
                predicted_match = df_value_matches[value][pred]
                result.loc[i, col] = (predicted_match.start_char, predicted_match.end_char)

        result["Name_Matched"] = [[x.char_span for x in df_name_matches[row["Name"]]] for _, row in df.iterrows()]
        return result

    def get_matches(self, df, line_text):
        df_name_matches = {row["Name"]:  self.match_name(row["Name"], line_text, debug_info=str(row)) 
                            for _, row in df.iterrows()}
        df_col_matches = {col: self.match_col(col, line_text) for col in df.columns[1:]}
        df_values = {row[col] for _, row in df.iterrows() for col in df.columns[1:] if row[col] != ""}
        df_value_matches = {value: self.match_value(value, line_text) for value in df_values}

        all_matches = list(
            itertools.chain.from_iterable(itertools.chain.from_iterable(x.values())
                                        for x in (df_col_matches, df_name_matches, df_value_matches)))
        Match.set_boundaries(all_matches, line_text)
        return df_name_matches, df_col_matches, df_value_matches

    def get_name_aliases(self, value):
        result = [value]

        if value in TEAM_ALIASES:
            return result + TEAM_ALIASES[value]

        # Add dots for abbreviated first name
        # Remove dots for abbreviated first name
        # Only use last name
        # Remove Jr., Sr., III
        sub_names = value.split(" ")
        if sub_names[0] == sub_names[0].upper() and "." not in sub_names[0]:
            result.append(".".join(sub_names[0]) + ". " + " ".join(sub_names[1:]))
        if sub_names[0] == sub_names[0].upper() and "." in sub_names[0]:
            result.append(sub_names[0].replace(".", "") + " " + " ".join(sub_names[1:]))
        if len(sub_names) > 1:
            result.append(" ".join(sub_names[1:]))
        if sub_names[-1] in remove_suffixes:
            result.extend([" ".join(r.split(" ")[:-1]) for r in result])
        if value == "Giannis Antetokounmpo":
            result.extend(["Giannis Antetenkoumpo", "Antetenkoumpo"])
        if value == "Jonas Valanciunas":
            result.extend(["Jonas Valancunius", "Valancunius"])
        if value == "Noah Vonleh":
            result.extend(["Noah Venloe", "Venloe"])
        if value == "Kobe Bryant":
            result.extend(["Kobe"])
        if value == "Kawhi Leonard":
            result.extend(["Kawhi"])
        return tuple(result)


    def match_name(self, name, text, debug_info=""):
        aliases = self.get_name_aliases(name)
        matches = self.match_value(name, text, no_match_strategy="ignore", aliases=aliases)

        if matches:
            return matches
        for dist in range(1, 3):
            for alias in aliases:
                matches = fuzzysearch.find_near_matches(alias, text, max_l_dist=dist)
                if matches:
                    return Match.from_list(matches)
        logger.warn(f"Couldn't match name '{name}' in '{text}'. ({debug_info})")
        return []

    def match_col(self, col, text):
        aliases = (col, ) + COL_ALIAS[col]
        matches = self.match_value(col, text, aliases=aliases, no_match_strategy="ignore")
        return matches


    def match_value(self, value, text, no_match_strategy="warning", debug_info="", aliases=None):
        matches = []
        aliases = aliases or (value, ) + NUMBER_WORDS.get(value, ())
        for alias in aliases:
            regex = re.compile(rf"\b{alias}\b")
            matches.extend(regex.finditer(text))
        if not matches and no_match_strategy == "warning":
            logger.warn(f"Couldn't match value '{value}' in '{text}'. ({debug_info})")
        return Match.from_list(matches)

    def compute_features(self, text, value, name, col, value_matches, name_matches, col_matches, team_matches):
        infc, infw, infs = len(text), len(text.split()), len(text.split("."))
        this_value_matches = value_matches[value]
        this_name_matches = name_matches[name]
        this_col_matches = col_matches[col]
        all_team_matches = [y for x in team_matches.values() for y in x]
        other_value_matches = [x for v in value_matches.keys() - {value} for x in value_matches[v]]
        other_name_matches = [x for n in name_matches.keys() - {name} for x in name_matches[n]]
        other_col_matches = [x for c in col_matches.keys() - {col} for x in col_matches[c]]
        result = []

        for value_match in sorted(this_value_matches, key=lambda x: x.start_char):
            # dist value name match
            ncb = min([x for x in (value_match.start_char - m.end_char + 1 for m in this_name_matches) if x >= 0] + [infc])
            nwb = min([x for x in (value_match.start_word - m.end_word + 1 for m in this_name_matches) if x >= 0] + [infw])
            nsb = min([x for x in (value_match.start_sent - m.end_sent + 1 for m in this_name_matches) if x >= 0] + [infs])

            nce = min([x for x in (m.start_char - value_match.end_char + 1 for m in this_name_matches) if x >= 0] + [infc])
            nwe = min([x for x in (m.start_word - value_match.end_word + 1 for m in this_name_matches) if x >= 0] + [infw])
            nse = min([x for x in (m.start_sent - value_match.end_sent + 1 for m in this_name_matches) if x >= 0] + [infs])

            # dist value col match
            ccb = min([x for x in (value_match.start_char - m.end_char + 1 for m in this_col_matches) if x >= 0] + [infc])
            cwb = min([x for x in (value_match.start_word - m.end_word + 1 for m in this_col_matches) if x >= 0] + [infw])
            csb = min([x for x in (value_match.start_sent - m.end_sent + 1 for m in this_col_matches) if x >= 0] + [infs])

            cce = min([x for x in (m.start_char - value_match.end_char + 1 for m in this_col_matches) if x >= 0] + [infc])
            cwe = min([x for x in (m.start_word - value_match.end_word + 1 for m in this_col_matches) if x >= 0] + [infw])
            cse = min([x for x in (m.start_sent - value_match.end_sent + 1 for m in this_col_matches) if x >= 0] + [infs])

            # num names in between
            ndb = len({x for x in (value_match.start_char - m.end_char + 1 for m in other_name_matches) if 0 <= x < ncb})
            nde = len({x for x in (m.start_char - value_match.end_char + 1 for m in other_name_matches) if 0 <= x < nce})
            nvb = len({x for x in (value_match.start_char - m.end_char + 1 for m in other_value_matches) if 0 <= x < ncb})
            nve = len({x for x in (m.start_char - value_match.end_char + 1 for m in other_value_matches) if 0 <= x < nce})

            # part of value pair?
            pair_delimiters = ("for", "-", "out", "of")
            ip1 = any(text[value_match.char_span[1]:].strip().startswith(x) and
                      re.split("\W", text[value_match.char_span[1]:].strip()[len(x):].strip())[0].isnumeric()
                      for x in pair_delimiters)
            ip2 = any(text[:value_match.char_span[0]].strip().endswith(x) and 
                      re.split("\W", text[:value_match.char_span[0]].strip()[:-len(x)].strip())[-1].isnumeric()
                      for x in pair_delimiters)

            # team matches
            tcb = min([x for x in (value_match.start_char - m.end_char + 1 for m in all_team_matches) if x >= 0] + [infc])
            twb = min([x for x in (value_match.start_word - m.end_word + 1 for m in all_team_matches) if x >= 0] + [infw])
            tsb = min([x for x in (value_match.start_sent - m.end_sent + 1 for m in all_team_matches) if x >= 0] + [infs])

            tce = min([x for x in (m.start_char - value_match.end_char + 1 for m in all_team_matches) if x >= 0] + [infc])
            twe = min([x for x in (m.start_word - value_match.end_word + 1 for m in all_team_matches) if x >= 0] + [infw])
            tse = min([x for x in (m.start_sent - value_match.end_sent + 1 for m in all_team_matches) if x >= 0] + [infs])

            # num cols in between
            cdb = len({x for x in (value_match.start_char - m.end_char + 1 for m in other_col_matches) if 0 <= x < ccb})
            cde = len({x for x in (m.start_char - value_match.end_char + 1 for m in other_col_matches) if 0 <= x < cce})
            cvb = len({x for x in (value_match.start_char - m.end_char + 1 for m in other_value_matches) if 0 <= x < ccb})
            cve = len({x for x in (m.start_char - value_match.end_char + 1 for m in other_value_matches) if 0 <= x < cce})

            result.append(
                (ncb, nwb, nsb, nce, nwe, nse, ccb, cwb, csb, cce, cwe, cse, ndb, nde, nvb, nve, cdb, cde, cvb, cve,
                 ip1, ip2, tcb, twb, tsb, tce, twe, tse)
                + COL_FEATURES[col]
            )
        return np.array(result)


class AlignerTrainer():
    def __init__(self, data_dir, cache_dir):
        self.cache_dir = cache_dir
        self.aligner = Aligner(data_dir, cache_dir)
        self.dataset = Rotowire("train", data_dir=data_dir)
        self.split = "train"

    def print_matches(self, string, **matches):
        markers = {}
        for color, c_matches in matches.items():
            for i, match in enumerate(c_matches):
                markers[match.start_char] = getattr(bcolors, color.upper())
                if color == "okgreen":
                    markers[match.start_char] += f"({i})"
                if match.end_char not in markers:
                    markers[match.end_char] = bcolors.ENDC
        for pos, col in sorted(markers.items(), reverse=True):
            string = string[:pos] + col + string[pos:]
        print(string)
    
    def store_features(self, split, all_features, all_labels, all_indexes, all_cols):
        group = np.hstack([i * np.ones_like(y, dtype=int) for i, y in enumerate(all_labels)])
        indexes = np.hstack([i * np.ones_like(y, dtype=int) for i, y in zip(all_indexes, all_labels)])
        cols = np.hstack([COLS.index(c) * np.ones_like(y, dtype=int) for c, y in zip(all_cols, all_labels)])
        labels = np.hstack(all_labels)
        features = np.vstack(all_features)
        stacked = np.hstack((features, indexes.reshape(-1, 1), cols.reshape(-1, 1),
                            group.reshape(-1, 1), labels.reshape(-1, 1)))
        np.save(self.cache_dir / split, stacked)
        return stacked

    def train_model(self, stacked_features, split):
        features = stacked_features[:, :-4]
        labels = stacked_features[:, -1]
        group = stacked_features[:, -2].astype(int)
        model.fit(features, labels, group)
        joblib.dump(model, self.cache_dir / (split + ".clf"))

    def load_cached(self, split):
        all_features, all_labels, all_indexes, all_cols = [], [], [], []
        col_counts = {k: 0 for k in COL_ALIAS.keys()}
        path = self.cache_dir / (split + ".npy")
        if path.exists():
            stacked_features = np.load(path)
            features = stacked_features[:, :-4].astype(int)
            labels = stacked_features[:, -1].astype(int)
            group = stacked_features[:, -2].astype(int)
            cols = stacked_features[:, -3].astype(int)
            indexes = stacked_features[:, -4].astype(int)

            for g in range(group.max() + 1):
                selector = group == g
                all_features.append(features[selector].tolist())
                all_labels.append(labels[selector].tolist())
                all_indexes.append(next(iter(indexes[selector])))
                all_cols.append(next(iter(COLS[x] for x in cols[selector])))

            for c in all_cols:
                col_counts[c] += 1
        return all_features, all_labels, all_indexes, all_cols, col_counts

    def run_interactive_alignment_training(self):
        all_features, all_labels, all_indexes, all_cols, col_counts = self.load_cached(self.dataset.split)
        skip_test = set(zip(all_indexes, all_cols))

        for i, team_df, player_df, line_text in self.dataset:

            for is_team_df, df in ((True, team_df), (False, player_df)):
                df_name_matches, df_col_matches, df_value_matches = self.aligner.get_matches(df, line_text)

                if is_team_df:
                    team_df_name_matches = df_name_matches

                for _, row in df.iterrows():
                    for col in df.columns[1:]:

                        this_col_count = col_counts.get(col, 0)
                        min_col_count = min(col_counts.values())
                        value = row[col]
                        if not value or this_col_count > min_col_count + 10 or (i, col) in skip_test:
                            continue

                        col_matches = df_col_matches[col]
                        name_matches = df_name_matches[row["Name"]]
                        value_matches = df_value_matches[value]

                        if len(value_matches) <= 1:
                            continue

                        print(col_counts, "\n")
                        self.print_matches(line_text, okgreen=value_matches, okblue=name_matches, warning=col_matches)

                        value_features = self.aligner.compute_features(line_text, value, row["Name"], col,
                                                                    df_value_matches, df_name_matches, df_col_matches,
                                                                    team_df_name_matches)
                        pred = model.predict(value_features) if i > 0 else 0
                        y = input(f"Choose correct value for {row['Name']} {value} {col} ({pred}) > ")
                        labels = np.zeros(len(df_value_matches[value]))
                        if y.isnumeric():
                            labels[int(y)] = 1
                        print(value_features)
                        print(labels)

                        all_features.append(value_features)
                        all_labels.append(labels)
                        all_indexes.append(i)
                        all_cols.append(col)

                        col_counts[col] = this_col_count + 1

                stacked_features = self.store_features(self.split, all_features, all_labels, all_indexes, all_cols)
                self.train_model(stacked_features, self.split)
