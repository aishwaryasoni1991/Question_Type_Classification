import os
from os import path


#REL_WORDS_DIR = path.join(path.dirname( "data/rel_words"))
##REL_WORDS_DIR =  "./data/rel_words" 
#
def build_word_lists():
    #build word lists from related words data 
    print 5

    #word_list_files = listdir(REL_WORDS_DIR)
    word_lists = {}
    a = 'data/rel_words'
    listing = os.listdir(a)
    print listing
    for wlf in listing:
        print "current file is: " + wlf
        with open(path.join(a,wlf),'r') as f:
            print 'in open'
            word_lists[wlf] = [word.strip().lower() for word in f.readlines()]
    print word_lists
    return word_lists
    

#print path.join(REL_WORDS_DIR)
build_word_lists()

 



#with open('C://','r+') as word_list_files:
#        for wlf in word_list_files:
#            print wlf
