from collections import Counter
import numpy as np
import math
import random

"""
CS 4120, Spring 2024
Homework 3 - starter code
"""

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"


# UTILITY FUNCTIONS

def create_ngrams(tokens: list, n: int) -> list:
    """Creates n-grams for the given token sequence.
    Args:
      tokens (list): a list of tokens as strings
      n (int): the length of n-grams to create

    Returns:
      list: list of tuples of strings, each tuple being one of the individual n-grams
    """
    # STUDENTS IMPLEMENT
    # the index (of original list) at which the current n-gram will start
    start_idx = 0
    ngrams = []
    while start_idx < len(tokens) - (n - 1):
        next_ngram = []
        # the index (of ngrams list) of the current n-gram we're making
        ngram_counter = 0
        while ngram_counter < n:
            next_ngram.append(tokens[ngram_counter + start_idx])
            ngram_counter += 1
        ngrams.append(tuple(next_ngram))
        start_idx += 1
    return ngrams


def read_file(path: str) -> list:
    """
    Reads the contents of a file in line by line.
    Args:
      path (str): the location of the file to read

    Returns:
      list: list of strings, the contents of the file
    """
    # PROVIDED
    f = open(path, "r", encoding="utf-8")
    contents = f.readlines()
    f.close()
    return contents


def tokenize_line(line: str, ngram: int,
                  by_char: bool = True,
                  sentence_begin: str = SENTENCE_BEGIN,
                  sentence_end: str = SENTENCE_END):
    """
    Tokenize a single string. Glue on the appropriate number of 
    sentence begin tokens and sentence end tokens (ngram - 1), except
    for the case when ngram == 1, when there will be one sentence begin
    and one sentence end token.
    Args:
      line (str): text to tokenize
      ngram (int): ngram preparation number
      by_char (bool): default value True, if True, tokenize by character, if
        False, tokenize by whitespace
      sentence_begin (str): sentence begin token value
      sentence_end (str): sentence end token value

    Returns:
      list of strings - a single line tokenized
    """
    # PROVIDED
    inner_pieces = None
    if by_char:
        inner_pieces = list(line)
    else:
        # otherwise split on white space
        inner_pieces = line.split()

    if ngram == 1:
        tokens = [sentence_begin] + inner_pieces + [sentence_end]
    else:
        tokens = ([sentence_begin] * (ngram - 1)) + \
            inner_pieces + ([sentence_end] * (ngram - 1))
    # always count the unigrams
    return tokens


def tokenize(data: list, ngram: int,
             by_char: bool = True,
             sentence_begin: str = SENTENCE_BEGIN,
             sentence_end: str = SENTENCE_END):
    """
    Tokenize each line in a list of strings. Glue on the appropriate number of 
    sentence begin tokens and sentence end tokens (ngram - 1), except
    for the case when ngram == 1, when there will be one sentence begin
    and one sentence end token.
    Args:
      data (list): list of strings to tokenize
      ngram (int): ngram preparation number
      by_char (bool): default value True, if True, tokenize by character, if
        False, tokenize by whitespace
      sentence_begin (str): sentence begin token value
      sentence_end (str): sentence end token value

    Returns:
      list of strings - all lines tokenized as one large list
    """
    # PROVIDED
    total = []
    # also glue on sentence begin and end items
    for line in data:
        line = line.strip()
        # skip empty lines
        if len(line) == 0:
            continue
        tokens = tokenize_line(line, ngram, by_char,
                               sentence_begin, sentence_end)
        total += tokens
    return total


class LanguageModel:

    def __init__(self, n_gram):
        """Initializes an untrained LanguageModel
        Args:
          n_gram (int): the n-gram order of the language model to create
        """
        # STUDENTS IMPLEMENT
        self.n = n_gram
        self.berp = tokenize(read_file(
            "training_files/berp-training.txt"), self.n)
        self.digits = tokenize(read_file(
            "training_files/digits.txt"), self.n)
        self.iamsam = tokenize(read_file(
            "training_files/iamsam.txt"), self.n)
        self.iamsam2 = tokenize(read_file(
            "training_files/iamsam2.txt"), self.n)
        self.unknowns_mixed = tokenize(read_file(
            "training_files/unknowns_mixed.txt"), self.n)
        self.unknowns = tokenize(read_file(
            "training_files/unknowns.txt"), self.n)
        self.all_training_files = [
            self.berp, self.digits, self.iamsam, self.iamsam2, self.unknowns, self.unknowns_mixed]

        self.ngrams = []
        self.unigrams = []
        self.n_minus_1_grams = []

        # will be assigned during training
        self.vocab_size = 0
        self.total_word_count = 0

    def train(self, tokens: list, verbose: bool = False) -> None:
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Args:
          tokens (list): tokenized data to be trained on as a single list
          verbose (bool): default value False, to be used to turn on/off debugging prints
        """
        # STUDENTS IMPLEMENT

        # replacing single-occurrence words with UNK
        print("TRAINING")
        toks_count = Counter(tokens)
        for x in tokens:
            if toks_count[x] == 1:
                tokens[tokens.index(x)] = UNK
        # creating ngrams
        self.ngrams += create_ngrams(tokens, self.n)
        self.n_minus_1_grams += create_ngrams(tokens, self.n - 1)
        self.unigrams += create_ngrams(tokens, 1)
        # vocab size is how many keys are in unigrams
        self.vocab_size = len(list(Counter(self.unigrams).keys()))
        self.total_word_count = len(self.unigrams)
        if verbose:
            print('first 50 ngrams', Counter(self.ngrams))
            print('first 50 unigrams', Counter(self.unigrams))
            print('vocab size (|V|)', self.vocab_size)
            print('total word count (N)', self.total_word_count)

    def score(self, sentence_tokens: list, verbose: bool = False) -> float:
        """Calculates the probability score for a given string representing a single sequence of tokens.
        Args:
          sentence_tokens (list): a tokenized sequence to be scored by this model

        Returns:
          float: the probability value of the given tokens for this model
        """
        # print("SCORING")
        input_toks = sentence_tokens
        # the count of each WORD in self.ngrams
        unigram_count = Counter(self.unigrams)
        # replace all unseen words with <UNK>
        for i in range(0, len(input_toks)):
            if unigram_count[(input_toks[i],)] == 0:
                input_toks[i] = UNK
        score = 1   # overall score to return
        denom = len(self.unigrams) + self.vocab_size
        if self.n == 1:   # Unigram case
            for tok in input_toks:
                multiplier = (unigram_count[(tok,)] + 1) / denom
                score *= multiplier
        else:
            # the count of each NGRAM in self.ngrams
            ngram_count = Counter(self.ngrams)
            # the count of each (n-1)gram in self.nminus1_grams
            n_minus_1_gram_count = Counter(self.n_minus_1_grams)

            hi_pointer = 0    # the idx of farthest ahead token to look at
            lo_pointer = 0    # the idx of farthest back token to look at
            while hi_pointer < len(input_toks):
                # P(A | B) = P(A & B) / P(B)
                # can be length 1 .. n
                A_B = tuple(input_toks[lo_pointer:hi_pointer + 1])
                # can be length 1 .. n-1
                B = tuple(input_toks[lo_pointer:hi_pointer])
                A_B_count = ngram_count[A_B]
                B_count = n_minus_1_gram_count[B]
                if len(B) == 0:   # means it's P(<s>), so just skip it (multiply by 1)
                    hi_pointer += 1
                    lo_pointer = hi_pointer - \
                        (self.n - 1) if hi_pointer - (self.n - 1) >= 0 else 0
                    continue
                elif len(B) == 1:   # means it's P(x|<s>), so the denominator is count(<s>)
                    B_count = unigram_count[(SENTENCE_BEGIN,)]
                # means it's P(x|<s>), numerator will be of length n-1
                elif len(A_B) == self.n - 1:
                    A_B_count = n_minus_1_gram_count[A_B]
                # calculate the score multiplier, using Laplace smoothing
                score *= ((A_B_count + 1) / (B_count + self.vocab_size))
                hi_pointer += 1
                # Markov Independence means our history is only N long
                lo_pointer = hi_pointer - \
                    (self.n - 1) if hi_pointer - (self.n - 1) >= 0 else 0
        return score

    def generate_sentence(self, verbose: bool = False) -> list:
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          list: the generated sentence as a list of tokens
        """
        print("GENERATING")
        return_sen = [SENTENCE_BEGIN]
        next_token = None
        self.scores_dict = {}
        while next_token != SENTENCE_END:
            if self.n == 1:   # Unigram case
                scores = self.get_scores(Counter(self.ngrams), [])
                next_token = random.choices(
                    self.ngrams, weights=scores, k=1)[0][0]
            else:   # Ngram case
                # get the last (n-1) items from the list
                # this will be length n-1 or shorter if its one of the first n words generated
                prev_tokens = return_sen[-(self.n-1):]
                # get all words where the first n-1 elements of its ngram match prev_tokens
                matching_words = [x for x in set(
                    self.ngrams) if x[:len(prev_tokens)] == tuple(prev_tokens)]
                # get the probabilities for these possibilities (to be used as weights)
                matching_ngrams_counter = Counter(matching_words)
                scores = self.get_scores(Counter(matching_ngrams_counter), [])
                next_token = random.choices(
                    matching_words, weights=scores, k=1)[0][-1]
            return_sen.append(next_token)
            if len(return_sen) > 20:  # for the sake of time
                return return_sen
        return return_sen

    # only calls score once per unique ngram, then appends that the appropriate number of times
    # made this function because generate_sentence() was taking an extremely long time
    def get_scores(self, counter: Counter, scores: list) -> list:
        for ngram in list(counter.keys()):
            # don't calculate the score of the same ngram twice
            try:
                score_to_append = self.scores_dict[ngram]
            except KeyError:  # this is a new ngram we haven't scored
                score_to_append = self.score(ngram)
                self.scores_dict[ngram] = score_to_append
            for i in range(0, counter[ngram]):
                scores.append(score_to_append)
        return scores

    def generate(self, n: int) -> list:
        """Generates n sentences from a trained language model using the Shannon technique.
        Args:
          n (int): the number of sentences to generate

        Returns:
          list: a list containing lists of strings, one per generated sentence
        """
        # PROVIDED
        return [self.generate_sentence() for i in range(n)]

    def perplexity(self, sequence: list) -> float:
        """Calculates the perplexity score for a given sequence of tokens.
        Args:
          sequence (list): a tokenized sequence to be evaluated for perplexity by this model

        Returns:
          float: the perplexity value of the given sequence for this model
        """
        # product of P(w_i | w_1 ... w_i-1)
        probability_score_product = self.score(sequence)
        # inverse probability formula
        perplexity_score = (1/probability_score_product) ** (1/self.vocab_size)
        return perplexity_score


# not required
if __name__ == '__main__':
    my_lm = LanguageModel(1)
    # for x in my_lm.all_training_files[:2]:
    #     my_lm.train(x, verbose=False)
    my_lm.train(my_lm.unknowns, verbose=True)
    print(my_lm.generate_sentence(verbose=True))
