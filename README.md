# Unscrambler 

Unscrambler is an n-gram based model to address an NLP challenge on 
data correction. This was done in a weekend as a result of a 
hiring process by part of an NLP-based company. 

The following statements will detail a bit more about the problem, 
the requirements, the approach taken, and some analysis done at the 
time. This document ends up summarizing possible improvements.


### Requirements

The `unscrambler` is developed in Python 3.7.3 and has as dependencies the 
following:

- nltk
- numpy
- dill (similar to pickle, but stores lambda expressions)

## Problem Statement

Given two documents, a clean one (`original_train.txt`) and a corrupted one 
(`scrambled_train.txt`), the goal is to create a system that is capable to 
recover corrupted files, that are generated from the same distribution as the 
clean file. The difference between the two files is the order of the 
sentence's elements. 

To this end, this project, named `unscrambler`, focus primarily on the 
creation of **Markovian Models** (or N-grams) as language models upon 
which it is then applied a strategy for decoding the sequence that is 
most likely to be in the correct order.


## Approach 

We split the training dataset (`original_train.txt` and `scrambled_train.txt`) 
according to a 80/20 split for training and validation, respectively. Then, 
we use `nltk` to build a probabilistic language model based on a Markov model of 
order `n`, also called N-grams. After training the model, we use this model to 
select the token with highest chance of being the correct one. This is choice 
is done according to a greedy decoding strategy. After decoding each sentence, 
we use `nltk.bleu.corpus_bleu` to estimate the performance of each model. In 
the end, we create in the folder `predictions` the results of applying the best 
models we found to the `test_scrambled.txt`.

For performance reasons, we also implement the `HeuristicsLanguageModel` which 
combines different heuristics (collected based on the data observations) and 
allow to select only the most promising candidates. Additionally and as an 
alternative to the `greedy` strategy, we have also implemented a recursive 
version of `beam search` decoding strategy, which expanded the `k` (default to 
2) most promising tokens and selected the one which would provide better results. 


## Results 

Regarding the data, we compared the overall token frequencies both for the 
training and validation data and since they follow more or less the same 
distribution, they can be seen as samples of the same distribution. 

Tokens distribution in Training Data (first 8000 sentences) 
![](https://gitlab.com/PastelBelem8/ai-research-challenge-2/raw/assets/imgs/train_data_term_freq.png "Training Data Tokens distribution") 

Tokens distribution in Validation Data (last 2000 sentences)
![](https://gitlab.com/PastelBelem8/ai-research-challenge-2/raw/assets/imgs/train_data_term_freq.png "Validation Data Tokens distribution")

Overall the token distributions are similar between the two datasets, thus 
strengthening our assumption that both samples could have been drawn from the 
same distribution.


Table with results... 

|        |         |
| ------ | ------- | 

## Main Thoughts

There's much to improve of course and even if I don't get selected for the next 
stage I would appreciate discussing this project with someone, so that I can 
learn with this experiment. 

So here's a wish list of what I would do had I had the chance: 

### Software-related Backlog
- Improve code performance (most of the code was implemented ad-hoc and using 
lists)
- Make well-defined tests to verify the code is performing as expected
- Improve beam-search implementation 


### AI-related 
- Research and test other approaches other than Markov Models (e.g. explore 
NNs for language model, explore skip-grams)
- Fine tune the heuristics weights (perhaps using a genetic algorithm for 
finding a close to optimal solution)
- Use results from Named Entity recognition in other English corpora to 
improve the `belongs_to_entity` heuristic. 
- Have more time to fine-tune the preprocessing (e.g, lowercase, punctuation, 
...)
- ...
