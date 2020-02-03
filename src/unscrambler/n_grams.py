import nltk
from nltk.lm import MLE, Laplace, KneserNeyInterpolated
import nltk.translate.bleu_score as b

from unscrambler.models import LanguageModel, HeuristicsLanguageModel, RandomModel
import unscrambler.utils as utils


def main():
    print("-" * 80)

    split = 0.80
    model, order = MLE, 4

    # Files Constants
    base_folder = '../data'
    original_filepath = f'{base_folder}/train_original.txt'
    original_sentences = utils.read_sentences(original_filepath)

    scrambled_filepath = f'{base_folder}/train_scrambled.txt'
    scrambled_sentences = utils.read_sentences(scrambled_filepath)

    # Split into train-val sets
    train_test_split = int(split * len(original_sentences))
    train_sentences = original_sentences[:train_test_split]
    correct_val_sentences = original_sentences[train_test_split:]
    scrambled_val_sentences = scrambled_sentences[train_test_split:]

    base_model = {
        "estimator": MLE,
        "order": 4,
        "strategy": "greedy",
        # "tokenize": nltk.tokenize.word_tokenize,
        "transformation": utils.transform_token,
        # "smoothing_function": b.SmoothingFunction().method2,
    }
    model = HeuristicsLanguageModel(**base_model)
    model.fit(train_sentences)
    predicted_val_sentences = [model.predict(s) for s in scrambled_val_sentences]
    print("Score:", model.score(correct_val_sentences, predicted_val_sentences))


if __name__ == '__main__':
    main()