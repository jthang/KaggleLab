import numpy
import re

from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import fit_ovo
import nltk

class StatelessTransform:
    """
    Base class for all transformations
    """
    def fit(self, X, y=None):
        return self

class ExtractText(StatelessTransform):
    """
    Extracts phrase text from datapoint class
    """
    def __init__(self, lowercase=False):
        """
        Lowercase the phrases
        """
        self.lowercase = lowercase

    def transform(self, X):
        """
        Get the lowercased words, returns tokenized and separated by single space " "
        """
        it = (" ".join(nltk.word_tokenize(datapoint.phrase)) for datapoint in X)
        if self.lowercase:
            return [x.lower() for x in it]
        return list(it)

class ReplaceText(StatelessTransform):
    def __init__(self, replacements):
        """
        Replacements is a list of (from, to) tuples of strings
        """
        self.rdict = dict(replacements)
        self.pat = re.compile("|".join(re.escape(origin) for orginin, _ in replacements))

    def transform(self, X):
        """
        X is a list of `str` instances
        Returns a list of `str` instances with replacements applied
        """
        if not self.rdict:
            return X
        return [self.pat.sub(self._repl_fun, x) for x in X]

    def _repl_fun(self, match):
        return self.rdict[match.group()]

class MapToSynsets(StatelessTransform):
    """
    Replaces words with Wordnet synsets
    """
    def transform(self, X):
        return [self._text_to_synsets(x) for x in X]

    def _text_to_synsets(self, text):
        result = []
        for word in text.split():
            ss = nltk.wordnet.wordnet.synsets(word)
            result.extend(str(s) for s in ss if ".n." not in str(s))
        return " ".join(result)

class Densifier(StatelessTransform):
    """
    Converts sparse matrix to numpy array
    """
    def transform(self, X, y=None):
        return X.todense()

class ClassifierOvOAsFeatures:
    "Dimensionality reduction - bag of words feature set into features"
    def fit(self, X, y):
        self.classifiers = fit_ovo(SGDClassifier(), X, numpy.array(y), n_jobs=4)[0]
        return self

    def transform(self, X, y=None):
        xs = [clf.decision_function(X).reshape(-1, 1) for clf in self.classifiers]
        return numpy.hstack(xs)


