import re
import fuzzysearch
import spacy

nlp_model = spacy.load("en_core_web_sm")

class Match():
    def __init__(self, match):
        if isinstance(match, fuzzysearch.common.Match):
            self.start_char = match.start
            self.end_char = match.end
            self.matched = match.matched
        else:
            self.start_char, self.end_char = match.span()
            self.matched = match.group(0)
        self.start_word = None
        self.end_word = None
        self.start_sent = None
        self.end_sent = None
        self.orig_match = match

    @property
    def char_span(self):
        return self.start_char, self.end_char

    @property
    def word_span(self):
        return self.start_word, self.end_word

    @property
    def sent_span(self):
        return self.start_sent, self.end_sent

    def set_word_id(self, boundaries):
        self.start_word = boundaries[self.start_char]
        self.end_word = boundaries[self.end_char - 1] + 1

    def set_sentence_id(self, boundaries):
        self.start_sent = boundaries[self.start_char]
        self.end_sent = boundaries[self.end_char - 1] + 1

    @staticmethod
    def set_boundaries(matches, text):
        word_boundaries = [w.span() for w in re.compile(r"\b\w+\b").finditer(text)]
        sentence_boundaries = [(s.start_char, s.end_char) for s in nlp_model(text).sents]

        word_boundaries_dict = Match.boundaries_to_dict(word_boundaries)
        sentence_boundaries_dict = Match.boundaries_to_dict(sentence_boundaries)

        for match in matches:
            match.set_word_id(word_boundaries_dict)
            match.set_sentence_id(sentence_boundaries_dict)

    @staticmethod
    def boundaries_to_dict(boundaries):
        current_id = 0
        result = {}
        next_word_id = None
        for i in range(boundaries[-1][-1]):
            if current_id is not None:
                _, current_end = boundaries[current_id]
                if i < current_end:
                    result[i] = current_id
                else:
                    result[i] = -1
                    next_word_id = current_id + 1
                    current_id = None
            else:
                next_start, _ = boundaries[next_word_id]
                if i < next_start:
                    result[i] = -1
                else:
                    result[i] = next_word_id
                    current_id = next_word_id
                    next_word_id = None
        return result

    @staticmethod
    def from_list(match_list, do_filter=True):
        result = []
        matches = [Match(m) for m in match_list]
        i = 0
        if do_filter:
            for match in  sorted(matches, key=lambda x: (x.start_char, -x.end_char)):
                if match.start_char >= i:
                    result.append(match)
                    i = match.end_char
        return result

    def __repr__(self):
        return (
            f"<Match object; char_span=({self.start_char}, {self.end_char}), word_span=({self.start_word}, "
            f"{self.end_word}), sent_span=({self.start_sent}, {self.end_sent}), matched={self.matched}>"
        )
