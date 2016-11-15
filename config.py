# Configu ration for nltk Stanford interface (doesn't work very well)
import os
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from feature_extractor import features

NER_DIR = os.path.join(os.path.dirname(__file__), "stanford-ner-2012-11-11")
NER_JAR = os.path.join(NER_DIR, "stanford-ner.jar")
NER_MODEL = "english.muc.7class.distsim.crf.ser.gz"
NER_MODEL = "english.conll.4class.distsim.crf.ser.gz"
NER_MODEL_PATH = os.path.join(NER_DIR, "classifiers", NER_MODEL)


tagger = StanfordNERTagger(NER_MODEL_PATH, NER_JAR)
tokenized_text = word_tokenize("my name is Rohan")
print tagger.tag(tokenized_text)


print features.RelatedWordVectorizer()
#word_lists={}
#
#with open("classes.txt",'r') as f:
#    print [word.strip().lower() for word in f.readlines()]
#            
#            
#f=open("classes.txt",'r')
#print [word.strip().lower() for word in f.readlines()]
#print f.readlines()

#
#print NER_DIR
#print NER_MODEL_PATH
#print NER_JAR      