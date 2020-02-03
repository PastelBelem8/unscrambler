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
![](https://github.com/PastelBelem8/unscrambler/raw/assets/imgs/train_data_term_freq.png "Training Data Tokens distribution") 

Tokens distribution in Validation Data (last 2000 sentences)
![](https://github.com/PastelBelem8/unscrambler/raw/assets/imgs/train_data_term_freq.png "Validation Data Tokens distribution")

Overall the token distributions are similar between the two datasets, thus 
strengthening our assumption that both samples could have been drawn from the 
same distribution.


Table with results... 

| Bleu   | Model   | Hyperparameters  | Strategy (if applicable)  | tokenization | punctuation | lowercase | numbers |
| ------ | ------- | ---------------- | ------------------------- | ------------ | ----------- | --------- | ------- |
|0.019465321| Random |seed: 101      |           |split   |no         |no       |<NUM>  |
|0.032996461| Random w/ Heuristics (all)    |seed: 101    |   |split   |no         |no       |<NUM>  |
|0.032996461| Random w/ Heuristics (1)      |seed: 101     |split   |no         |no       |<NUM>  |
|0.032996461| Random w/ Heuristics (2)      |seed: 101     |split   |no         |no       |<NUM>  |
|0.032996461| Random w/ Heuristics (3)      |seed: 101     |split   |no         |no       |<NUM>  |
|0.032996461| Random w/ Heuristics (4)      |seed: 101     |split   |no         |no       |<NUM>  |
|0.032996461| Random w/ Heuristics (5)      |seed: 101     |split   |no         |no       |<NUM>  |
|0.03456929 |Language Model|MLE, 1            |greedy     |split   |no         |no       |<NUM>  |
|0.036862993|      |MLE, 2                                       |           |split   |no         |no       |<NUM>  |
|0.032853874|      |MLE, 3                                       |           |split   |no         |no       |<NUM>  |
|0.03299167 |      |MLE, 4                                       |           |split   |no         |no       |<NUM>  |
|0.033295661|      |MLE, 5                                       |           |split   |no         |no       |<NUM>  |
|0.033295661|      |MLE, 6                                       |           |split   |no         |no       |<NUM>  |
|0.033295661|      |MLE,7                                        |           |split   |no         |no       |<NUM>  |
|           |Language Model w/ Heuristics|MLE, 1                 |greedy     |split   |no         |no       |<NUM>  |
|           |      |MLE, 2                                       |greedy     |split   |no         |no       |<NUM>  |
|           |      |MLE, 3                                       |greedy     |split   |no         |no       |<NUM>  |
|           |      |MLE, 4                                       |greedy     |split   |no         |no       |<NUM>  |
|           |      |MLE, 5                                       |greedy     |split   |no         |no       |<NUM>  |
|           |      |MLE, 6                                       |greedy     |split   |no         |no       |<NUM>  |
|           |      |MLE,7                                        |greedy     |split   |no         |no       |<NUM>  |
|0.033161876|Language Model|MLE, 1                               |beam, k = 2|split   |no         |no       |<NUM>  |
|0.03456929 |      |MLE, 1                                       |beam, k = 3|split   |no         |no       |<NUM>  |
|           |      |MLE, 2                                       |beam, k = 2|        |           |         |       |
|           |      |MLE, 3                                       |beam, k = 2|        |           |         |       |
|           |      |MLE, 4                                       |beam, k = 2|        |           |         |       |
|           |      |MLE, 5                                       |beam, k = 2|        |           |         |       |
|           |      |MLE, 6                                       |beam, k = 2|        |           |         |       |
|           |      |MLE,7                                        |beam, k = 2|        |           |         |       |
|0.03456929 |Language Model w/ Heuristics|Laplace, 1             |greedy     |split   |no         |no       |<NUM>  |


Due to time constraints this table was not completed... 


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
- Take a closer look to the data, and try to look up some patterns (perhaps, 
reverse the order of the sentence)

