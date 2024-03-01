from datetime import datetime
import logging
from pathlib import Path
from eleet_pretrain.datasets.pretraining.python_processing.utils import rand_term, shorten_uri

logger = logging.getLogger(__name__)


TEXT_GEN_TEMPLATE_FILE = Path(__file__).parents[3] / "NeuralDB" / "dataset-construction" / "configs" / "generate_v1.5.json"  # noqa
DATE_PATTERNS = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:00Z",
    "%Y-%m-%dT%H:00:00Z",
    "%Y-%m-%dT00:00:00Z",
    "%Y-%m-00T00:00:00Z",
    "%Y-00-00T00:00:00Z"
]

def convert_date(date_str):
    orig_date_str = date_str
    suffix = ""
    if date_str.startswith("+"):
        date_str = date_str[1:]
    elif date_str.startswith("-"):
        date_str = date_str[1:]
        suffix = " bc"

    date = None
    for pattern in DATE_PATTERNS[::-1]:
        try:
            date = datetime.strptime(date_str, pattern)
            break
        except ValueError:
            continue

    if date is None:  # year probably too large
        result = date_str.split("-")[0] + suffix
        logger.warn(f"Couldn't parse date {orig_date_str}. Assumed reason: Year too large. Returning {result}.")
        return result

    date_formatted = list()
    if date.hour or date.minute:
        date_formatted.append(date.strftime("%H:%M"))
    if date.day:
        date_formatted.append(date.strftime("%d"))
    if date.month:
        date_formatted.append(date.strftime("%B"))
    if date.year:
        date_formatted.append(date.strftime("%Y") + suffix)
    return " ".join(date_formatted)


def get_value_converter(key, labels, aliases, rng):
    _value_converters = {
        "default": {
            "time": lambda x: convert_date(x["time"]),
            "wikibase-entityid": lambda x: labels.get(x["id"], x["id"]),
            "globecoordinate": lambda x: str(x["longitude"]) + ", " + str(x["latitude"]),
            "quantity": lambda x: x["amount"] + labels.get(shorten_uri(x["unit"]), ""),
            "monolingualtext": lambda x: x["text"] if x["language"] == "en" else "",
            "string": lambda x: x
        },
        "rand_term": {
            "time": lambda x: convert_date(x["time"]),
            "wikibase-entityid": lambda x: rand_term(
                aliases=aliases, rng=rng, entity_id=x["id"], labels=labels),
            "globecoordinate": lambda x: str(x["longitude"]) + ", " + str(x["latitude"]),
            "quantity": lambda x: x["amount"] + labels.get(shorten_uri(x["unit"]), ""),
            "monolingualtext": lambda x: x["text"] if x["language"] == "en" else "",
            "string": lambda x: x
        },
        "uri": {
            "time": lambda x: f"{x['time']}^^http://www.w3.org/2001/XMLSchema#dateTime",
            "globecoordinate": lambda x: None,
            "monolingualtext": lambda x: None,  # TODO maybe consider those in the future.
            "quantity": lambda x: None,
            "wikibase-entityid": lambda x: f"http://www.wikidata.org/entity/{x['id']}",
            "string": lambda x: None
        }
    }
    return lambda x: _value_converters[key][x[0]](x[1])


class PseudoDict():
    def __init__(self, func):
        self.func = func

    def get(self, key, default=None):
        try:
            r = self.func(key)
            if r is None:
                return default
            return r
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key):
        r = self.func(key)
        if r is None:
            raise KeyError(key)
        return r

    def __contains__(self, key):
        return self.func(key) is not None


class WikidataQualifier():
    def __init__(self, obj, labels, aliases, rng):
        self.obj = obj
        self.labels = labels
        self.aliases = aliases
        self.rng = rng

    def iter(self, convert="default", filter_none=True):
        for o in self.obj:
            if o is not None:
                o =  get_value_converter(convert, self.labels, self.aliases, self.rng)(o)
            if o is not None or not filter_none:
                yield o

    def __len__(self):
        return len(self.obj)


class WikidataQualifiers():
    def __init__(self, obj, labels, aliases, rng):
        self.obj = obj
        self.labels = labels
        self.aliases = aliases
        self.rng = rng
    
    def items(self):
        for k in self.keys():
            yield k, self[k]

    def __getitem__(self, key):
        return WikidataQualifier(self.obj[key], self.labels, self.aliases, self.rng)

    def keys(self):
        return self.obj.keys()

    def __iter__(self):
        return iter(self.__keys__())

    def __len__(self):
        return len(self.obj)

class WikidataProperty():
    def __init__(self, obj, labels, aliases, rng):
        self.obj = obj
        self.labels = labels
        self.aliases = aliases
        self.rng = rng

    def iter(self, convert="default", qualifiers=False, filter_none=False):
        for o, q in self.obj:
            if o is not None:
                o = get_value_converter(convert, self.labels, self.aliases, self.rng)(o)
            if o is None and filter_none:
                continue
            if qualifiers:
                yield o, WikidataQualifiers(q, self.labels, self.aliases, self.rng)
            else:
                yield o

    def __len__(self):
        return len(self.obj)

class WikidataProperties():
    def __init__(self, obj, labels, aliases, rng):
        self.obj = obj
        self.labels = labels
        self.aliases = aliases
        self.rng = rng
    
    def __getitem__(self, key):
        return WikidataProperty(self.obj[key], self.labels, self.aliases, self.rng)

    def keys(self):
        return self.obj.keys()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.obj)

    def items(self):
        for k in self.keys():
            yield k, self[k]
