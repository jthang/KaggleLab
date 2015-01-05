import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data',
            'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data',
             'testData.tsv'), header=0, delimiter="\t", quoting=3)

    print 'The first review is:'
    print train["review"][0]

    raw_input("Press Enter to continue")

    print "Cleaning and parsing movie reviews...\n"
    clean_train_reviews = []
    for i in xrange(0, len(train['review'])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train['review'][i], True)))

    print "Creating bag of words...\n"

    vectorizer = CountVectorizer(analyzer = 'word',
                                tokenizer = None,
                                preprocessor = None,
                                stop_words = None,
                                max_features = 5000)

    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()

    print "Training Random Forest \n"

    rf = RandomForestClassifier(n_estimators=100)
    clf = rf.fit(train_data_features, train['sentiment'])

    print "Clean and parse test set \n"
    clean_test_reviews = []
    for i in xrange(0, len(test['review'])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test['review'][i], True)))

    test_data_features = vectorizer.fit_transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    print "Predicting test labels \n"
    result = clf.predict(test_data_features)

    output = pd.DataFrame(data={'id':test['id'], 'sentiment':result})
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'rf.csv'), index=False, quoting=3)
    print "Submission file created"



