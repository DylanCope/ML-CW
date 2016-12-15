from glob import glob
import pickle
from functools import partial
from itertools import chain, combinations, product
from operator import methodcaller
from email.parser import Parser
import re
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import sys

def lmap( *args, **kwargs ):
    return map( *args, **kwargs )

def unpackTraining( index, emails, y ):
    ''' Unpacks training data from a KFold split '''
    train_index, test_index = index
    return emails[ train_index ], y[ train_index ]

def unpackTesting( index, emails, y ):
    ''' Unpacks testing data from a KFold split '''
    train_index, test_index = index
    return emails[ test_index ], y[ test_index ]

def transformEmail( featureWords, email ):
    '''
        Returns a boolean vector describing feature
        words that appear in an email.
    '''
    return lmap( partial( contains, email ), featureWords )

def transformEmails( featureWords, preprocess, emails ):
    '''
        Params:
            featureWords: a list of strings
            preprocess: a function from strings to strings
            emails: a list of strings
        Returns:
            A matrix of boolean row vectors, where each row
            corresponds to an email and each column corresponds
            to a feature word.
    '''
    processed = lmap( preprocess, emails )
    transform = partial( transformEmail, featureWords )
    return np.matrix(lmap( transform, processed ))

def simpleModel( emails, labels,
                 getFeatureWords = simpleFeatureWords,
                 preprocess = simplePreprocess,
                 hyperparams = { 'alpha' : 1, 'class_prior' : [0.5, 0.5] }  ):
    '''
        Constructs a simple Naive Bayes Model.
        Params:
            emails: a list of strings
            labels: a list of integers that are either 0 or 1
            getFeatureWords: a function that maps a list of
            strings to a list of strings
            preprocess: a function that maps a string to a string
            hyperparams: a dictionary that maps an 'alpha' to a number
            and a 'class_prior' to a list of two numbers.
        Returns:
            clf: A trained Multinomial classifier
            transform: a function that maps a list of strings to
            a data matrix.

    '''
    processed = lmap( preprocess, emails )
    featureWords = getFeatureWords( processed )
    transform = partial( transformEmails, featureWords, preprocess )
    clf = MultinomialNB( **hyperparams )
    X = transform( emails )
    clf.fit( X, labels )
    return clf, transform

def crossValScore( model, n_splits = 5 ):
    '''
        Params:
            model: This parameter is a function that takes a data matrix and
            a labelling vector and returns a classifier and transformation
            function. The classifier is an object that has a score method that
            takes a data matrix and a labelling vector and returns a number.
            The transformation function takes a list of totally unprocessed
            emails and returns a data matrix.
        Returns:
            An array of scores corresponding to each CV fold.
    '''
    emails, y = getData()
    split = KFold( n_splits = n_splits,
                   shuffle = True,
                   random_state = 0 ).split( emails )
    scores = []
    for index in split:
        clf, transform = model( *unpackTraining( index, emails, y ) )
        emails_test, y_test = unpackTesting( index, emails, y )
        X_test = transform( emails_test )
        scores += [ clf.score( X_test, y_test ) ]

    return np.array(scores)


def multipartContent( email ):
    payload = email.get_payload()
    content = [ multipartContent(x) if x.is_multipart() else x.get_payload() for x in payload ]
    return reduce( lambda x, y : x + ' ' + y, content, '' )

def contentOnly( e ):
    parser = Parser()
    email = parser.parsestr(e)
    content = email.get_payload()
    if email.is_multipart():
        return multipartContent( email )
    return content

def contentAndRelevantHeaders( text ):
    content = contentOnly( text )
    parser = Parser()
    email = parser.parsestr( text )
    headers = []
    for k, v in email.items():
        if k in [ 'Subject', 'To', 'From'  ]:
            headers.append( v )
    return reduce( lambda x, y : '%s %s' % (y, x), headers + [content], '' )

def processSymbols( text ):
    text = re.sub( r'__+', ' multiscore ', text )
    text = re.sub( r'\*(\*)+', ' multistar ', text )
    text = re.sub( r'\!(\!)+', ' multibang ', text )
    text = re.sub( r'\?(\?)+', ' multiques ', text )
    text = re.sub( r'\.(\.)+', ' multidots ', text )
    text = re.sub( r'#(#)+', ' multihash ', text )
    text = re.sub( r'-(-)+', ' multidash ', text )
    text = text.replace( '$', ' money ' )
    text = text.replace( '', ' money ' )
    text = text.replace( '!', ' bang ' )
    text = text.replace( '%', ' percent ' )
    text = text.replace( '&', ' and ' )
    for symbol in [ '.', ',', '@', '_', '<', '>', '{', '}' '\\', '/', '#', '^', '=', \
                    '(', ')', '"', ':', ';', '~', '+', '*', '?', '[', ']', '\%' ]:
        text = text.replace( symbol, ' ' )
    return text

def hasNumbers( string ):
    return any( c.isdigit() for c in string )

def isNumber( string ):
    return string.replace('.', '').isdigit()

def isAscii(s):
    return all(ord(c) < 128 for c in s)

def processNumerics( text ):
    words = text.split()
    words = [ 'num'      if isNumber(w)    else w for w in words ]
    words = [ 'alphanum' if hasNumbers(w)  else w for w in words ]
    words = [ 'nonascii' if not isAscii(w) else w for w in words ]
    return reduce( lambda x, y : '%s %s' % (x, y), words, '' )

def processHyperlinks( text ):
    text = re.sub(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', ' hyperlink ', text )
    text = re.sub( r'^https?:\/\/.*[\r\n]*', ' hyperlink ', text )
    text = re.sub( r'<.*?>', ' htmltag ', text )
    return text

def isAllCaps( string ):
    return all(map( methodcaller('isupper'), string ))

def processCapitals( text ):
    words = text.split()
    words = [ ' allcaps %s ' % w if isAllCaps(w) else w for w in words ]
    words = map( methodcaller('lower'), words )
    return reduce( lambda x, y : '%s %s' % (x, y), words, '' )

def advancedPreprocess( email ):
    text = contentAndRelevantHeaders( email )
    text = processHyperlinks( text )
    text = processNumerics( text )
    text = processSymbols( text )
    text = processCapitals( text )
    words = [ w for w in text.split() if len(w) > 1 ]
    return reduce( lambda x, y : '%s %s' % (x, y), words, '' )

if __name__ == '__main__':
    filename = sys.argv[1]
    with open( filename ) as f:
        email = f.read()
    with open( './vocab.txt' ) as f:
        vocab = f.read().split()
    vectorizer = CountVectorizer( binary = True,
                                  preprocessor = advancedPreprocess,
                                  vocabulary = vocab )
    X = vectorizer.transform([ email ])
    clf = joblib.load( './classifier.dat' )
    print( { 1 : 'spam', 0 : 'ham' }[ clf.predict(X)[0] ] )
