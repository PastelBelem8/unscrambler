from nltk.corpus import stopwords

import string
import numpy as np


# -------------------------------------------------------------------
# Auxiliar functions
# -------------------------------------------------------------------
def read_all(filepath: str) -> str:
    """Real all the contents of the ``filepath``."""
    with open(filepath, 'r', encoding='utf-8', newline='\n') as f:
        return f.read()


def read_sentences(filepath: str, n: int=None) -> list:
    """Read the first n lines from ``filepath``."""
    lines = read_all(filepath).splitlines()
    return lines if n is None else lines[:n]


def write(contents: list, filepath: str):
    """Write the content in the specified ``filepath``."""
    with open(filepath, "w", encoding="utf-8", newline='\n') as f:
        f.write("\n".join(contents))


# -------------------------------------------------------------------
# Preprocessing functions
# -------------------------------------------------------------------
# Mapping table
punct_table = str.maketrans('', '', string.punctuation)
stopWords = set(stopwords.words('english'))


def transform_token(token: str, drop_punct: bool=False, lowercase: bool=False, tokenize_nums: bool=True) -> str:
    """Transforms ``token`` by removing the punctuation and lowering the case."""
    # Remove punctuation
    t = token.translate(punct_table) if drop_punct else token
    # Cast to lowercase
    t = t.lower() if lowercase else t
    # Handle numbers
    handle_numbers(t) if tokenize_nums else t
    return t


def is_capital(t: str) -> bool:
    """Determines whether a token is capitalized in the first case
    and all others are non-capitalized."""
    return t[0].isupper() and t[1:].islower()


def is_end_of_sentence(t: str) -> bool:
    """Determines whether the token is a end-of-sentence token."""
    return any(map(lambda m: t.endswith(m), ('.', '?', '!', '...')))


def is_stopword(t: str) -> bool:
    """Determines whether the token is a stopword."""
    return t in stopWords


def handle_numbers(t: str) -> str:
    """Tokenizes the tokens which represent numbers, placing <NUM> tokens
    instead."""
    if t.isdigit():
        return '<NUM>'
    elif any(map(str.isdigit, t.split('.'))):
        return '<NUM>'
    else:
        return t


def belongs_to_entity(context, t):
    """Checks whether the context requires other entity recognition"""
    # Naive approach for now
    return is_capital(context[-1]) and is_capital(t)


def get_top_n_tks(context: list, candidate_tks: list, n: int) -> list:
    """Computes the `n` most promising tokens for a certain context."""
    n_candidates = len(candidate_tks)
    if n_candidates < n:
        return n_candidates

    else:
        tks_score = []
        used_tks = len(context) / (len(context)+ n_candidates)

        for t in candidate_tks:
            score = 0

            if is_capital(t):
                # If at beginning of sentence -> give higher score to capital letters
                score += int(context == ['<s>'])

                # If previous word is part of Entity -> higher score to capital letters
                score += 0.5 * int(belongs_to_entity(context, t))

            # If many candidate tokens left -> give lower score to *end_of_sentence* tokens
            score +=  0.5 * np.log2(used_tks) if is_end_of_sentence(t) else 0

            # If it is a stop word -> give lower score as these are typically
            # favoured by probabilistic LM
            score -= 0.1 if int(is_stopword(t)) else 0

            tks_score.append(score)

    # Return index of top n tokens
    return sorted(range(n_candidates), key=lambda i: tks_score[i])[-n:]