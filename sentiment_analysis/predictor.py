from collections import defaultdict
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import accuracy_score

from transformations import (ExtractText, ReplaceText, MaptoSynsets, Densifier, ClassifierOvOasFeatures)
from inquirer_lex_transform import InquirerLexTransform

_valid_classifiers = {
    "sgd": SGDClassifier,
    "knn": KNeighborsClassifier,
    "svc": SVC,
    "randomforest": RandomForestClassifier
}


