""" Train a question classifier model for question answering """
from os import path,listdir
import sys
import re
import fileinput
from pprint import pprint
from time import time
#import argparse
import logging as log
import numpy
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import StratifiedKFold, LeaveOneOut
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import feature_extractor
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import configuration
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
#reload(sys)
#sys.setdefaultencoding("utf-8")

class Classifier:

    def __init__(self, init_data=None):
        self.data = init_data
       # self.model_file = model_file
        self.model = self.build_model()
        

    def build_model(self):
        """
        Create the model (pipeline) with any parameters. Using pipeline we can transform the data as we want and then 
        call the fit and predict method only once rather than calling it everytime for each transformation
        """
        # Pipeline is used to chain multiple estimators into one. It stores a list of tuples of key value pairs
        # FeatureUnion takes a list of transformer object
        print "in model now"
        print feature_extractor.features.TagVectorizer()
        #print features.RelatedWordVectorizer()
        
        model = Pipeline([
            ('union', FeatureUnion([
                ('words', TfidfVectorizer(max_df=0.25, ngram_range=(1, 4),
                                          sublinear_tf=True, max_features=5000)),
                                          
                ('relword',feature_extractor.features.RelatedWordVectorizer(max_df=0.75, ngram_range=(1, 4),
                                                  sublinear_tf=True)),
                
                ('pos', feature_extractor.features.TagVectorizer(max_df=0.75, ngram_range=(1, 4),
                                       sublinear_tf=True)),
                 ('ner', feature_extractor.features.NERVectorizer(min_df= 0.5,ngram_range=(1, 4),
                                       sublinear_tf=True)),
            ])),
            
            ('clf', LinearSVC()),
           
        ])
       
        
        return model

    def train_model(self):
        """
        Train the model with extracted features from all the data

        For a sklearn pipeline example, see:
        http://scikit-learn.org/stable/auto_examples/grid_search_text_feature_extraction.html
        """
        log.debug("Training model...")
      #  print 'in train'
        self.model.fit(self.data.data, self.data.target)

    
    
    def save_model(self,filename):
        if filename=="":
            log.ERROR("Empty file name")
        print "in saving model " ,filename
        log.debug("Saving model to file: " + filename)      
        joblib.dump(self.model,path.join(configuration.MODEL_DIR,filename))
        
    def load_model(self,pklfile):
        if pklfile=="":
            log.ERROR("Empty file name")
        log.debug("Loading model from file: " + pklfile)
        self.model = joblib.load(pklfile)
        return self
    
    def predict(self, doc):
        """
        Predict the classification of a document with the trained model
        Returns the coarse and fine classes
        """
        qtype = self.model.predict([doc])
        #print "prediction is ",qtype
        return qtype
       # return qtype[0], qtype[1]
        
        #change the n_folds value =10

   
    def test_faq(self):
        X= self.data.data # list of questions
        #print 'X is ', X
        
        y = self.data.target # list of labels
        #print 'Y is  ', y
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, test_size=0.33)
        self.load_model(path.join(configuration.MODEL_DIR,"train.pkl"))
        test_pred = []
        for a in X_test:
            print "a is "
            print
            print a
            
            test_pred=test_pred.append(self.predict(a))
        result = self.model.score(test_pred, Y_test)
        b= accuracy_score(Y_test,test_pred)
        
        print("Accuracy: %.3f%%") % (result*100.0)
        print "b is ", b
            
        
        return 0
    
    
    def test_model(self,n_folds=1,leave_one_out=False): #n_folds=1
        """
        Test the model by cross-validating with Stratified k-folds

        """
        log.debug("Testing model ({} folds)".format(n_folds))
        
        X = self.data.data # list of questions
        #print 'X is ', X
        
        y = self.data.target # list of labels
        #print 'Y is  ', y
        
        avg_score = 0.0

        if leave_one_out:
            #print 'leave_one_out=True'
            cv = LeaveOneOut(len(y))
            print "cv is true",cv
            #print 'length of y is',len(y)
            #print 'cv.n is ',cv.n
        else:
            #print 'leave_one_out=False'
            cv = StratifiedKFold(y) 
            #print "cv is false ",cv
            """Stratification will ensure that the percentages of each class in your entire data 
            will be the same (or very close to) within each individual fold."""
            #print 'cv.y is ',cv.y

    # in every iteration, one label will be left out, starting from 0 location
        for train, test in cv:
            print train, test
            print 'x train ', X[train]
            print 'y train is ', y[train]
          
            model = self.build_model().fit(X[train], y[train])
            avg_score += model.score(X[test], y[test])
            print 
            print 'x test and y test'
            print X[test], y[test]

        if leave_one_out:
            avg_score /= len(y)
        else:
            avg_score /= n_folds

        #print("Average score: {}".format(avg_score))
        scores = cross_val_score(model, X, y, cv=cv) # X=  all the questions, y is the target
        print
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return avg_score

    
def classify_question_type(text):
    print "in the classify question function"
    clf = Classifier()
    clf.load_model(path.join(configuration.MODEL_DIR,"train.pkl"))

    #print
    qtype = clf.predict(text) 
     
    print qtype
    return qtype
    
    
def load_data(filenames, coarse=False):

    data = [] # data stores the actual question
    target = [] # target stores the coarse labels like HUM or NUM etc
    fine_target = [] # fine_target stores the fine labels like manners,food etc
    print "coarse is ",coarse
    if coarse:
        data_re = re.compile(r'(\w+):(\w+) (.+)')
        print 'line is '
        print data_re
    else:
        # removes the labels and considers only question
        data_re = re.compile(r'(\w+:\w+) (.+)')
        #print 'false coarse line is '
        #print data_re

    for line in fileinput.input(filenames):
        #line=line.decode('utf-8','ignore').encode("utf-8")
        d = data_re.match(line)        
        if not d:
            raise Exception("Invalid format in file {} at line {}"
                            .format(fileinput.filename(), fileinput.filelineno()))
        if coarse:
            target.append(d.group(1))
            #print 'target dgroup1 '
            #print target 
            fine_target.append(d.group(2))
            
            data.append(d.group(3))
            #print 'target dgroup3 '
            #print data
        else:
            target.append(d.group(1))
            data.append(d.group(2))
            

    return Bunch(
        data=numpy.array(data),
        target=numpy.array(target),
        target_names=set(target), #selects only the unique labels
    )



""" Model trained and pickled on coarse =False will predict only for coarse=Flase and not coarse = true and vicevera"""
    
if __name__ == "__main__":
    print datetime.now().time()
    
    #data = load_data("./data/train.txt",coarse=False)
    #test_data=load_data("./data/faq.txt")
    #print data # prints the Bunch() output

    
    #clf = Classifier(data)
    #clf.search_estimator_params()
    #        # clf.test_model(n_folds=10)
    
    
    #clf.train_model()
    #clf.test_faq()
    #clf.test_model(n_folds=2) #change value to 2
    # Plot Precision-Recall curve
    
    #clf.save_model("trainCoarse.pkl")
    
    #clf.load_model(path.join(configuration.MODEL_DIR,"train.pkl"))
    print
    print
    #clf.
    #clf=joblib.load(path.join(configuration.MODEL_DIR,"train.pkl"))
    #print clf
    classify_question_type(": Why are you requiring all this coursework? I am an experienced software engineer and shouldn't have to take these academic prerequisites.") 
    #clf.test_model(leave_one_out=True)
    #clf.predict("How do the Nazis justify the killings of jews in the Holocaust?") 
    print datetime.now().time()
         