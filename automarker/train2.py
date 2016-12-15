from glob import glob
import pickle
from functools import reduce, partial
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

def getEmails( folder = None, preprocess = lambda x : x ):
    '''
        Params:
            folder: a string representing a folder
            preprocess: a function that maps strings to strings
        Returns:
            A list where each item is an email in folder
    '''
    if folder:
        files = lmap( open, glob( '../%s/*.txt' % folder ) )
        emails = [ preprocess( f.read() ) for f in files ]
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
    emails, labels = getData()

    vecparams = { 'binary'        : True,
                  'preprocessor'  : advancedPreprocess,
                  'token_pattern' : r'\b\w\w+\b',
                  'strip_accents' : True,
                  'lowercase'     : True }

    vectorizer = CountVectorizer( **vecparams )
    vectorizer.fit( emails )
    X = vectorizer.transform( emails )

    hyperparams = { 'alpha' : 0.7, 'class_prior' : [0.9, 0.1] }
    clf = MultinomialNB( **hyperparams )
    clf.fit( X, labels )

    # printAccuracy(crossValScore( lambda es, ls : clf, vectorizer.transform ))
    joblib.dump( clf, 'classifier.dat' )
    with open( 'vocab.txt', 'w' ) as f:
        for w in vectorizer.get_feature_names():
            f.write( '%s\n' % w )
