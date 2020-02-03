from abc import ABC, abstractmethod
import numpy as np

# Evaluation
import unscrambler.evaluation as e
import nltk.translate.bleu_score as b

# Preprocessing
from nltk.lm.preprocessing import padded_everygram_pipeline
import math

# Model Persistence
import dill as pickle # dill includes the lambda functions when saving models
import unscrambler.utils as utils


class DecodingStrategy:
    def get_strategy(mode, **kwargs):
        if mode == 'greedy':
            return DecodingStrategy.greedy
        else:
            return DecodingStrategy.beam_search

    def greedy(elems: list, utility, *args, **kwargs):
        scores = list(map(utility, elems))
        return max(scores), np.argmax(scores)

    def beam_search(elems, utility, rollout, k: int=2):
        n_elems = len(elems)
        scores = list(map(utility, elems))

        if n_elems <= k:
            return max(scores), np.argmax(scores)
        else:
            top_k_elems = sorted(range(n_elems), key=lambda i: scores[i])[-k:]
            top_k_results = [rollout(k) for k in top_k_elems]

            return sorted(top_k_results, key=lambda x: (x[0], x[1]))[-1]


class BaseModel(ABC):
    """Skeleton class for all models devised to solve the scramble AI
    challenge."""
    def __init__(self, **kwargs):
        self._tokenize = kwargs.get('tokenize', str.split)
        self._transform_tkn = kwargs.get('transformation', lambda x: x)

        smooth_f = kwargs.get('smoothing_function',
                              b.SmoothingFunction(epsilon=1e-8).method1)
        self._score_function = lambda t, p: \
            e.evaluate_bleu(t, p, smoothing_function=smooth_f)

        strategy_mode = kwargs.get('strategy', 'greedy')
        self._strategy = DecodingStrategy.get_strategy(strategy_mode)

    @abstractmethod
    def fit(self, train_data):
        raise NotImplementedError

    def score(self, true: list, pred: list) -> float:
        """Score the model's predictions."""
        return self._score_function(true, pred)

    def select_candidate_tokens(self, prev_tks, candidate_tokens):
        return candidate_tokens

    def select_best_token(self, prev_tks, candidate_tokens):
        raise NotImplementedError

    def predict_next_token(self, prev_tks: list, available_tks: list) -> int:
        """Predict the index of the next token in ``available_tks`` based on
        previous tokens."""
        if len(available_tks) == 1:
            return 0

        candidate_tokens = self.select_candidate_tokens(prev_tks, available_tks)
        best_score, best_tkn = self.select_best_token(prev_tks, candidate_tokens)
        return best_tkn

    def predict(self, sentence: str) -> tuple:
        """Predicts an ordered sentence."""
        # The punctuation is correct, only the order is different
        original_tks = sentence.split()
        available_tks = map(self._transform_tkn, self._tokenize(sentence))

        # Remove empty strs
        available_tks = list(filter(lambda t: t, available_tks))
        final_sent = ""
        final_tks = ["<s>"]

        while len(available_tks) > 0:
            next_token = self.predict_next_token(final_tks, available_tks)
            final_tks += [available_tks.pop(next_token)]
            final_sent += f"{original_tks.pop(next_token)} "

        return final_sent[:-1]

    def dump(self, filename):
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(self, f)


class RandomModel(BaseModel):
    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        np.random.seed(seed)

    def fit(self, train_data):
        # There's really nothing to fit, is there? (:
        pass

    def predict_next_token(prev_tks: list, available_tks: list) -> int:
        """Predict the index of the next token in ``available_tks`` based on
        previous tokens."""
        if len(available_tks) == 1:
            return 0
        else:
            return np.random.randint(0, len(available_tks))


class LanguageModel(BaseModel):
    def __init__(self, estimator, order: int, **kwargs):
        """Creates and train a language model based on an n-order Markov
        model for text, where `n` is the order of the Markovian model and
        the text is the set of sentences.

        :param lm: class of the Language Model to create.
        :param order: the order of the Markov model to create (uni-grams
        are 0 order, bi-grams 1 order, ...)
        :param sentences: the sentences of the training set
        :return: the trained model
        """
        super().__init__(**kwargs)
        self.estimator = estimator
        self.order = order
        self.model = estimator(order)

    def fit(self, train_data: str):
        """Fits the model for the list of sentences provided in ``train_data``.

        :param train_data: list of sentences to consider for training
        """
        sents_tkns = []
        for sent in train_data:
            tokens = self._tokenize(sent)
            tokens = [self._transform_tkn(t) for t in tokens if t]
            sents_tkns.append(tokens)

        ngrams, padded_sents = padded_everygram_pipeline(self.order, sents_tkns)
        self.model.fit(ngrams, padded_sents)

    def _get_context(self, tks):
        context_size = min(self.order - 1, len(tks))
        return tks[-context_size:]

    def select_best_token(self, prev_tks, candidate_tokens):
        def rollout(tk):
            temp_cand_tks = candidate_tokens.copy()
            tk = temp_cand_tks.pop(tk)

            temp_prev_tks = prev_tks.copy()
            temp_prev_tks.append(tk)
            return self.select_best_token(temp_prev_tks, temp_cand_tks)

        context = self._get_context(prev_tks)
        score_function = lambda t: self.model.logscore(t, context)
        best_tkn = self._strategy(candidate_tokens, score_function, rollout)
        return best_tkn


class HeuristicsLanguageModel(LanguageModel):
    def __init__(self, max_candidates='sqrt', **kwargs):
        super().__init__(**kwargs)

        # Determine how many candidates to select in each time
        if max_candidates is None:
            self._get_max_candidates = len
        elif max_candidates == 'sqrt':
            self._get_max_candidates =  lambda ts: math.ceil(math.sqrt(len(ts)))
        elif max_candidates == 'halve':
            self._get_max_candidates = lambda ts: len(ts) // 2
        else:
            self._get_max_candidates = lambda ts: max(0, min(math.ceil(max_candidates * len(ts)), len(ts)))

    def select_candidate_tokens(self, prev_tks, candidate_tokens):
        if len(candidate_tokens) == 1:
            candidate = [0]

        top_n = self._get_max_candidates(candidate_tokens)
        # else apply heuristics
        candidates = utils.get_top_n_tks(prev_tks, candidate_tokens, top_n)
        return [candidate_tokens[i] for i in candidates]