""" Feature extractors for question classification """
from os import path, listdir
from itertools import chain, product

import numpy as np
from nltk import pos_tag
#from nltk.tag.stanford import NERTagger
from nltk.tag import StanfordNERTagger
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import ner

import config
#from inquire import config


#REL_WORDS_DIR = path.join(path.dirname(__file__), "data/rel_words")
#REL_WORDS_DIR =  "./data/rel_words" 

# create a dict of all the related words

def build_word_lists():
    #build word lists from related words data 
    print 5
    word_lists = {}
    a = 'data/rel_words'
    listing = listdir(a)
    #print listing
    for wlf in listing:
        #print "current file isl: " + wlf
        f=open(path.join(a,wlf),'r')
        #with open(path.join(a,wlf),'r') as f:
        print 'in open'
        word_lists[wlf] = [word.strip().lower() for word in f.readlines()]
    #print word_lists
    return word_lists

class TagVectorizer(TfidfVectorizer):
    def __init__(self, tags_only=False, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super(TagVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=False,
            dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf)

        self.tags_only = tags_only

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""

        preprocess = self.build_preprocessor()
        stop_words = self.get_stop_words()
        tokenizer = self.build_tokenizer()
        tokenize = lambda doc: tokenizer(preprocess(self.decode(doc)))
        
        # nltk pos_tag returns tuples of the form (word, TAG)
        if self.tags_only:
            # We are just interested in the tags here, so we do tagged_tuple[1]
            get_tags = lambda doc: [t[1] for t in pos_tag(tokenize(doc))]

        else:
            get_tags = lambda doc: list(chain.from_iterable(pos_tag(tokenize(doc))))
        return lambda doc: self._word_ngrams(get_tags(doc), stop_words)


class NERVectorizer(TfidfVectorizer):

    def __init__(self, tags_only=True, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super(NERVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=False,
            dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf)

        
        self.tags_only=tags_only
        self.tagger = StanfordNERTagger(config.NER_MODEL_PATH, config.NER_JAR, encoding=self.encoding)
        print "in NER ",tags_only
        print self.tags_only
        #self.tagger = ner.SocketNER(host='localhost', port='9191', output_format='slashTags')
        print "heelok asoks ",self.tagger

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        preprocess = self.build_preprocessor()
        tokenizer = self.build_tokenizer()
        tokenize = lambda doc: tokenizer(preprocess(self.decode(doc)))
        
        print "in build analyzer of NER "

        # get_tags = lambda doc: [tag for tag in self.tagger.get_entities(doc).iterkeys()]

        #if self.tags_only:
        #    
        #    get_tags = lambda doc: [t[0] for t in self.tagger.get_entities(doc)]
        #else:
        #    get_tags = lambda doc: list(chain.from_iterable(self.tagger.get_entities(doc)))
        
        

        if self.tags_only:
            
            get_tags = lambda doc: [t[1] for t in self.tagger.tag(tokenize(doc))]
            print "self tags true ",get_tags("What is the most local place in paris ?")
              
        else:
           
            get_tags = lambda doc: list(chain.from_iterable(self.tagger.tag(tokenize(doc))))
            print "self tags false ",get_tags
        
        #print lambda doc: self._word_ngrams(get_tags(doc))
        return lambda doc: self._word_ngrams(get_tags(doc))

class RelatedWordVectorizer(TfidfVectorizer):
    # just create a new string of "rel_word" tags and pass it into a TfidfVectorizer
    def __init__(self, tags_only=False, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super(RelatedWordVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=False,
            dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf)

        print "in wordvect"
        print
        print self.preprocessor
        self.word_lists = build_word_lists()

    
    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        print 'in rel build analyizer'

        preprocess = self.build_preprocessor()
        tokenize = self.build_tokenizer()

        return lambda doc: self._word_ngrams(self.build_rel_word_string(
            tokenize(preprocess(self.decode(doc)))))

    def get_rel_word(self, word):
        print "word in get rel is ", word
        for rel, words in self.word_lists.iteritems():
            if word in words:
                return rel
        return ""

    def build_rel_word_string(self, doc):
        print "in build_rel word doc is ",doc
        related_words = ""
        for word in doc:
            rel_word = self.get_rel_word(word)
            if rel_word:
                related_words += rel_word + " "
        return related_words.strip()
   

   

