import nltk.translate.bleu_score as nltk_bleu


def evaluate_bleu(true_sents: list, pred_sents: list, **kwargs) -> float:
    references = [[l] for l in map(str.split, true_sents)]
    hypothesis = list(map(str.split, pred_sents))

    print("Evaluating BLEU score")
    return nltk_bleu.corpus_bleu(references, hypothesis, **kwargs)
