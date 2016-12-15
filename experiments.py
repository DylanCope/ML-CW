
# coding: utf-8

# # Spam Filter
#
# The objective of this is to create a classifier to distinguish between spam and non-spam (ham) emails.

# In[16]:

import os
import random
import collections
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
from itertools import chain
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from functools import reduce
import random
import nltk
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from sklearn.metrics import accuracy_score




pylab.rcParams['figure.figsize'] = (32.0, 24.0)
pylab.rcParams['font.size'] = 24


# In[17]:

import sys
sys.version


# In[26]:

def processEmail( email ):
    return [ w for w in email.split() if w.isalpha() ]

def getEmails( folder ):
    '''
        Given a folder to retrieve from,
        returns a list where each item is a list of the words in an email
    '''
    emails = []
    file_list = os.listdir( folder )
    for a_file in file_list:
        with open(folder + '/' + a_file, 'r',encoding='utf-8', errors='ignore') as f:
            emails.append( f.read() )

    return list(map( processEmail, emails ))

def getData( feature_extract ):

    spamEmails = getEmails( 'spam' )
    hamEmails = getEmails( 'ham' )

    X = np.matrix(list(map( feature_extract, chain( spamEmails, hamEmails ) )))
    y = [ 1 ] * len( spamEmails ) + [ 0 ] * len( hamEmails )

    return X, y

def stemEmails(emails):
    ''''
    Takes the result of getEmails,
    returns the stemmed results according to Porter's algorithm
    '''
    porter_stemmer = PorterStemmer()
    portStem = lambda word : porter_stemmer.stem(word)
    stemmed = [[portStem(word) for word in email] for email in emails]
    return stemmed


def filterStopWords(emails):
    '''
    Takes the result of getEmails and removes stopwords
    and words with length of < 2
    '''
    stop_list = get_stop_words('english')
    validWord = lambda word : word.lower() not in stop_list and len(word) > 2

    filtered = [[ word for word in email if validWord(word)] for email in emails]
    return filtered



# ## Random Classification

# In[27]:

# dummy_clf = DummyClassifier( strategy = 'uniform', random_state = 0 )
# X, y = getData( lambda e : [1] )
# dummy_clf.fit( X, y )
# dummy_clf.score( X, y )


# In[28]:

# dummy_clf = DummyClassifier( strategy = 'most_frequent', random_state = 0 )
# X, y = getData( lambda e : [1] )
# dummy_clf.fit( X, y )
# dummy_clf.score( X, y )


# In[29]:

def probabilities( emails ):
    '''
        Given a list were each item is a list of words,
        returns a dictionary that map words to probabilties
    '''
    rm_dups = lambda l : list(set( l ))
    counter = collections.Counter( sum( map(rm_dups, emails), [] ) )
    return { k : v / len( emails ) for k, v in counter.items() }

def getMostProbable( dist, n ):
    '''
        Given a probability distribution and a number of instances to get,
        returns the n most probable instances
    '''
    items = list( dist.items() )
    items.sort( key = lambda x : -x[1] )
    return dict( items[ : n ] )

def plotHist( dist, title = '' ):
    '''
        Plots a probablity distribution as a histogram
    '''
    items = list( dist.items() )
    items.sort( key = lambda x : -x[1] )
    author_names = [ k for k, v in items ]
    author_counts = [ v for k, v in items ]

    # Plot histogram using matplotlib bar().
    indexes = np.arange(len(author_names))
    width = 0.7

    fig = plt.figure( figsize = (32.0, 7.0) )
    ax = fig.add_subplot( 111 )
    ax.bar( indexes, author_counts, width)
    ax.set_xticks( indexes + width * 0.5 )
    ax.set_xticklabels( author_names, rotation = 45 )
    ax.set_title( title )
    plt.show()

def histDifference( h1, h2 ):
    '''
        Given two probability distributions,
        returns a new distribution representing a key-wise difference
    '''
    return { k : h1[ k ] - h2[ k ] for k in h1.keys() if k in h2 }


# In[1]:




# In[40]:

spamProbs = probabilities( getEmails( 'spam' ) )
hamProbs  = probabilities( getEmails( 'ham'  ) )
# plotHist( getMostProbable( spamProbs, 30 ), 'Top 30 words in spam' )
# plotHist( getMostProbable( hamProbs, 30 ), 'Top 30 words in ham' )
#
# # Repeat for stemmed spam and ham
stemmedSpamProbs = probabilities( stemEmails( filterStopWords(getEmails( 'spam' ))) )
stemmedHamProbs  = probabilities( stemEmails( filterStopWords(getEmails( 'ham' ))) )
# plotHist( getMostProbable( stemmedSpamProbs, 30 ), 'Top 30 words in stemmed spam' )
# plotHist( getMostProbable( stemmedHamProbs, 30 ), 'Top 30 words in stemmed ham' )


# In[31]:



spamHamDiff = histDifference( spamProbs, hamProbs )
# plotHist( getMostProbable( spamHamDiff, 30 ), 'Top 30 words that positively identify spam' )

hamSpamDiff = histDifference( hamProbs, spamProbs )
# plotHist( getMostProbable( hamSpamDiff, 30 ), 'Top 30 words that positively identify ham' )

# Repeat for stemmed spam and ham
stemmedSpamHamDiff = histDifference( stemmedSpamProbs, stemmedHamProbs )
# plotHist( getMostProbable( stemmedSpamHamDiff, 30 ), 'Top 30 words that positively identify stemmed spam' )

stemmedHamSpamDiff = histDifference( stemmedHamProbs, stemmedSpamProbs )
# plotHist( getMostProbable( stemmedHamSpamDiff, 30 ), 'Top 30 words that positively identify stemmed ham' )


# In[34]:

n = 30
mostIdentifying = getMostProbable( dict( chain( hamSpamDiff.items(), spamHamDiff.items() ) ), n )
# plotHist( mostIdentifying, 'Top %i most identifying words across both classes' % n )

n = 30
stemmedMostIdentifying = getMostProbable( dict( chain( stemmedHamSpamDiff.items(), stemmedSpamHamDiff.items() ) ), n )
# plotHist( stemmedMostIdentifying, 'Top %i most identifying words across both stmmed classes' % n )


# In[32]:

def pickingMostIdentifying( n ):
    mostIdentifying = getMostProbable( dict( chain( hamSpamDiff.items(), spamHamDiff.items() ) ), n )
    featureWords = mostIdentifying.keys()
    X, y = getData( lambda e : [ w in e for w in featureWords ] )
    clf = BernoulliNB()
    return cross_val_score( clf, X, y, cv=5 ).mean()


# In[35]:

# fig = plt.figure( figsize = (32, 7) )
# ax = fig.add_subplot( 111 )
# xs = list(range( 5, 30 ))
# mapping = [ ( x, pickingMostIdentifying( x )) for x in xs ]
# ax.set_title( 'Performance of Naive Bayes Classifier' )
# ax.set_ylabel( 'Score' )
# ax.set_xlabel( 'Number of features' )
# ax.plot( xs, ys )
# plt.show()


# In[36]:

mostIdentifying = getMostProbable( dict( chain( hamSpamDiff.items(), spamHamDiff.items() ) ), 10 )
featureWords = mostIdentifying.keys()
X, y = getData( lambda e : [ w in e for w in featureWords ] )

# Repeat for stemmed version
stemmedMostIdentifying = getMostProbable( dict( chain( stemmedHamSpamDiff.items(), stemmedSpamHamDiff.items() ) ), 10)
stemmedFeatureWords = stemmedMostIdentifying.keys()
X_stemmed, y_stemmed = getData( lambda e : [ w in e for w in stemmedFeatureWords ] )





################################################################################

def compareClassifications(X, y):
    '''
        - Takes a dataset, fits the dataset to the classifier,
        then quantifies the Brier score loss(mean square value) before
        and after calibration
        - also plots reliability curves before and after Classification
    '''
    from sklearn.metrics import accuracy_score
    clf = MultinomialNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    prob_pos_clf = clf.predict_proba(X_test)[:,1]
    clf_score = brier_score_loss(y_test, prob_pos_clf)

    predicted, actual = compare(X_train, X_test, y_train, y_test, clf)
    print("Accuracy: %f" % accuracy)
    print("Brier score %f" % clf_score)
    print("predicted portion of positive class: %f" % predicted)
    print("actual portion of positive class: %f" % actual)


    # Naive-Bayes with isotonic calibration
    clf_isotonic = CalibratedClassifierCV(clf, cv='prefit', method='isotonic')
    clf_isotonic.fit(X_train, y_train)
    prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]
    clf_score = brier_score_loss(y_test, prob_pos_isotonic)

    predicted, actual = compare(X_train, X_test, y_train, y_test, clf_isotonic)
    print("After calibration")
    print("Accuracy: %f" % accuracy)
    print("Brier score %f" % clf_score)
    print("predicted portion of positive class: %f" % predicted)
    print("actual portion of positive class: %f" % actual)

    plotClassified(X_train, X_test, y_train, y_test, clf)
    plotClassified(X_train, X_test, y_train, y_test, clf_isotonic)





def plotClassified(X_train, X_test, y_train, y_test, clf):
    bin_width = 0.1
    clf.fit(X_train, y_train)
    # Take positive class probailites from the predicted matrix
    probabilities = [ p[0] for p in clf.predict_proba(X_test)]

    xs = []
    ys = []
    for i in [float(j) / 100 for j in range(10, 100, 1)]:

        numInstances = 0
        for p in probabilities:
            if p <= i + bin_width and p >= i - bin_width:
                numInstances += 1
        xs.append(i)
        ys.append(numInstances / len(probabilities))

    print(xs)
    print(ys)
    fig_size = [10,10]
    plt.rcParams["figure.figsize"] = fig_size
    plt.axis([0,1,0,1])
    plt.scatter(xs,ys)
    plt.plot([0,1],[0,1],linestyle="--")
    plt.plot(xs,ys)
    plt.xlabel("class probability")
    plt.ylabel("fraction of class")
    plt.show()


def compare(X_train, X_test, y_train, y_test, clf):

    clf.fit(X_train, y_train)
    # iterate through test dataset and find proportion of correctly identified instances
    averageProb = 0
    numPositiveInstances = 0
    actualNumPositiveInstances = 0

    for X_t, y_t in zip(X_test, y_test):
        val = clf.predict(X_t)[0]
        # prob is the probably that instance is in the positive class
        prob = clf.predict_proba(X_t)[0][1]

        averageProb += prob
        actualNumPositiveInstances += y_t
        numPositiveInstances += val

    averageProb = averageProb / len(y_test)
    ratioOfFound = actualNumPositiveInstances / len(y_test)
    return (averageProb, ratioOfFound)


# Take a random test sample from the dataset in order to demonstrate calibration of classifier
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#
# clf = MultinomialNB()
# scores = cross_val_score( clf, X, y, cv=5 )
# print("Accuracy without calibration : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# averageProb, ratioOfFound = compare(X_train, X_test, y_train, y_test, clf)
# print("average probability: %f" % averageProb)
# print("ratio of found: %f" % ratioOfFound)
#
# clf.fit(X_train, y_train)
# clf_score = brier_score_loss(y_test, clf.predict_proba(X_test)[:,1])
# print("Brier score loss: %f" % clf_score)
#
# plotClassified(X_train, X_test, y_train, y_test, clf)
#
# clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
# scores = cross_val_score( clf_isotonic, X, y, cv=5 )
# print("Accuracy with calibration : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# averageProb, ratioOfFound = compare(X_train, X_test, y_train, y_test, clf_isotonic)
# print("average probability: %f" % averageProb)
# print("ratio of found: %f" % ratioOfFound)
# clf_score = brier_score_loss(y_test, clf_isotonic.predict_proba(X_test)[:,1])
# print("Brier score loss: %f" % clf_score)
#
# plotClassified(X_train, X_test, y_train, y_test, clf_isotonic)

compareClassifications(X, y)
###############################################################################
###############################################################################

# split train, test for calibration
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)


# clf = MultinomialNB()
# clf.fit(X_train, y_train)  # GaussianNB itself does not support sample-weights
# prob_pos_clf = clf.predict_proba(X_test)[:, 1]
#
# print(list(prob_pos_clf))
# averageProb = sum(list(prob_pos_clf)) / len(list(prob_pos_clf))
# print("average probability: %f" % averageProb)
# prob = clf.predict_proba(X_test)[0][1]
# print("probability prior to calibration: %f" % prob)

# Gaussian Naive-Bayes with isotonic calibration
# clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
# clf_isotonic.fit(X_train, y_train)
# prob = clf_isotonic.predict_proba(X_test)[0][1]
# print("probability after calibration: %f" % prob)



# clf = MultinomialNB()
# scores = cross_val_score( clf, X, y, cv=5 )
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Accuracy achiebed using stemmed version
# print("Accuracies achieved using stemmed data")
# clf = BernoulliNB()
# stemmed_scores = cross_val_score( clf, X_stemmed, y_stemmed, cv=5 )
# print("Accuracy: %0.2f (+/- %0.2f)" % (stemmed_scores.mean(), stemmed_scores.std() * 2))

# In[ ]:
#
# clf = MultinomialNB()
# stemmed_scores = cross_val_score( clf, X_stemmed, y_stemmed, cv=5 )
# print("Accuracy: %0.2f (+/- %0.2f)" % (stemmed_scores.mean(), stemmed_scores.std() * 2))
