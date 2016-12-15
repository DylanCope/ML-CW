import sys
from train import *
from sklearn.externals.joblib import load

if __name__ == '__main__':
    filename = sys.argv[ 1 ]
    with open( filename ) as f:
        email = filter( isAscii, f.read() )
    clf = load( './classifier.dat' )
    print( clf.classify( email ) )
