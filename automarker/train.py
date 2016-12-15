import re
import sys
import pickle
import collections
import numpy as np

from glob                            import glob
from email.parser                    import Parser
from operator                        import methodcaller
from sklearn.externals               import joblib
from functools                       import reduce, partial
from sklearn.naive_bayes             import MultinomialNB
from itertools                       import chain, combinations, product
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation        import KFold

def getEmails( folder = None, preprocess = lambda x : x ):
    '''
        Params:
            folder: a string representing a folder
            preprocess: a function that maps strings to strings
        Returns:
            A list where each item is an email in folder
    '''
    if folder:
        files = map( open, glob( '../%s/*.txt' % folder ) )
        emails = [ preprocess(filter( isAscii, f.read() )) for f in files ]
        map( methodcaller('close'), files )
        return emails
    else:
        return getEmails( 'spam' ) + getEmails( 'ham' )

def getData():
    ''' Returns a list of unprocessed emails and corresponding labels. '''
    spam = getEmails( 'spam' )
    ham = getEmails( 'ham' )
    emails = np.array( spam + ham )
    y = np.array( len( spam ) * [1] + len( ham ) * [0] )
    return emails, y

def multipartContent( email ):
    ''' Recursively extract multipart email content '''
    payload = email.get_payload()
    content = [ multipartContent(x) if x.is_multipart() else x.get_payload() for x in payload ]
    return reduce( lambda x, y : x + ' ' + y, content, '' )

def contentOnly( e ):
    ''' Extract content from email '''
    parser = Parser()
    email = parser.parsestr(e)
    content = email.get_payload()
    if email.is_multipart():
        return multipartContent( email )
    return content

def contentAndRelevantHeaders( text ):
    ''' Extract content and some headers '''
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
    text = text.replace( '!', ' bang ' )
    text = text.replace( '%', ' percent ' )
    text = text.replace( '&', ' and ' )
    for symbol in [ '.', ',', '@', '_', '<', '>', '{', '}' '\\', '/', '#', '^', '=', '(', ')', '"', ':', ';', '~', '+', '*', '?', '[', ']', '\%' ]:
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

def getAllWords( emails ):
    '''
        Params: emails: a list of strings
        Returns: Set of strings that appear separated
            by whitespace in emails
    '''
    return set(sum( [ e.split() for e in emails ], [] ))

def probabilities( emails, words ):
    '''
        Given a list were each item is a list of words,
        returns a dictionary that map words to probabilties
    '''
    ws = sum( [ [ w for w in e.split() if w in words ] for e in emails ], [] )
    counter = collections.Counter( ws )
    return { k : v / len( ws ) for k, v in counter.items() }

def histDifference( h1, h2 ):
    '''
        Given two probability distributions,
        returns a new distribution representing a key-wise difference
    '''
    return { k : h1[ k ] - h2[ k ] for k in h1.keys() if k in h2 }

def getFeatureWords( emails, labels ):
    spam = [ e for e, l in zip( emails, labels ) if l == 1 ]
    ham  = [ e for e, l in zip( emails, labels ) if l == 0 ]
    words = getAllWords( emails )
    spamProbs = probabilities( spam, words )
    hamProbs  = probabilities( ham, words )
    spamHamDiff = histDifference( spamProbs, hamProbs )
    hamSpamDiff = histDifference( hamProbs, spamProbs )
    mostIdentifying = dict( ( k, max( abs(spamHamDiff[k]), abs(hamSpamDiff[k]) ) ) for k in chain( hamSpamDiff, spamHamDiff ) )

    return mostIdentifying

def unpackTraining( index, emails, y ):
    ''' Unpacks training data from a KFold split '''
    train_index, test_index = index
    return emails[ train_index ], y[ train_index ]

def unpackTesting( index, emails, y ):
    ''' Unpacks testing data from a KFold split '''
    train_index, test_index = index
    return emails[ test_index ], y[ test_index ]

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
    split = KFold( len( emails ),
                   n_folds = n_splits,
                   shuffle = True,
                   random_state = 0 )
    scores = []
    for index in split:
        clf, transform = model( *unpackTraining( index, emails, y ) )
        emails_test, y_test = unpackTesting( index, emails, y )
        X_test = transform( emails_test )
        s = clf.score( X_test, y_test )
        print('Accuracy: %0.3f' % s)
        scores += [ s ]

    return np.array(scores)

class WeightedMultinomialNB( MultinomialNB ):

    def __init__( self, weights, vectorizer, **kwargs ):
        super( WeightedMultinomialNB, self ).__init__( **kwargs )
        self.weights = np.array(weights)
        self.vectorizer = vectorizer

    def _update_feature_log_prob( self ):
        ''' Weight the feature logarithm probabilities '''
        super( WeightedMultinomialNB, self )._update_feature_log_prob()
        self.feature_log_prob_ += np.log(1 + self.weights)

    def classify( self, email ):
        ''' Returns 'spam' or 'ham' '''
        X = self.vectorizer.transform([ email ])
        predict = super( WeightedMultinomialNB, self ).predict
        return { 1 : 'spam', 0 : 'ham' }[ predict(X)[0] ]

def buildModel( emails, labels ):

    vecparams = {   'binary'        : True,
                    'preprocessor'  : advancedPreprocess,
                    'token_pattern' : r'\b\w\w+\b',
                    'strip_accents' : True,
                    'lowercase'     : True }
    hyperparams = { 'alpha'         : 0.7,
                    'class_prior'   : [0.8, 0.2] }

    emails, labels = getData()
    processed = map( advancedPreprocess, emails )
    featureWords = getFeatureWords( processed, labels )
    vocab = list( featureWords.keys() )
    weights = list( featureWords.values() )

    vectorizer = CountVectorizer( vocabulary = vocab, **vecparams )
    vectorizer.fit( emails )
    X = vectorizer.transform( emails )
    clf = WeightedMultinomialNB( weights, vectorizer, **hyperparams )
    clf.fit( X, labels )

    return clf, vectorizer.transform

if __name__ == '__main__':
    scores = crossValScore( buildModel )
    print('Mean accuracy: %0.3f (+/- %0.3f)' % (scores.mean(), scores.std()))
    clf, _ = buildModel( *getData() )
    joblib.dump( clf, 'classifier.dat' )
