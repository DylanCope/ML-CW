import urllib.request
import re
from functools import reduce

def getRelatedWords( word ):
    website = 'http://wordnetweb.princeton.edu/perl/webwn'
    webvars = 'sub=Search+WordNet&o2=&o0=&o8=1&o1=&o7=&o5=&o9=&o6=&o3=&o4=&h=000000000'
    url = '%s?s=%s&%s' % ( website, word, webvars )
    try:
        with urllib.request.urlopen( url ) as response:
            html = response.read().decode()
            html = re.sub( r'<.*?>', '', html )
            lines = [ re.sub( 'S: \(.*?\)', '', l ) for l in html.split( '\n' ) if l[:2] == 'S:' ]
            text = reduce( lambda x, y : '%s,%s' % (x, y), lines, '' )
            # text = text.replace( '\n', ',' )
            phrases = [ phrase.split() for phrase in text.split( ',' ) ]
            words = list(set([ p[0] for p in phrases if len(p) == 1 ]))
            return words
    except:
        pass
    return []

vocab = ['fire', 'force', 'coat']
for word, words in zip( vocab, map( getRelatedWords, vocab ) ):
    print(word, ' = ', words or [word])
