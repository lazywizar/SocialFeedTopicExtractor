#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:27:00 2018

@author: vkumar
"""
import os
import guidedlda.utils
import time
from six.moves import cPickle as pickle
import numpy as np
import guidedlda

import twitter

import sys, getopt
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
import enchant

#Global
HOME = '/Users/vkumar/MCS/CS410/Project/'
TOPIC_INDEX = {0: 'Sports', 1: 'Business', 2: 'Arts/Litrature', 3: 'Politics', 4: 'Misc'}

#Twitter App keys and secrets must be present as environment variables.
CONSUMER_KEY = os.environ.get("CONSUMER_KEY", "")
CONSUMER_SECRET = os.environ.get("CONSUMER_SECRET", "")
ACCESS_TOKEN_KEY = os.environ.get("ACCESS_TOKEN_KEY", "")
ACCESS_TOKEN_SECRET = os.environ.get("ACCESS_TOKEN_SECRET", "")

#### Create models
def train_model(dataset, vocab, seed_topic_list, model_output_path, n_topics=5, n_top_words=10):
    """Function takes a dataset and creates a new model based on the privided input dataset
            
    Args:
        dataset (dtm): Dataset in dtm format as guided by guidedlda.datasets.* 
        vocab (list(str)): Global vocab.
        seed_topic_list (array(list(str)): Seed topic keywords used for GuidedLDA.
        model_output_path (str): Path to the ooutput of trained model
        n_topics (int): Number of topics, default=5
        n_top_words (int): Number of top words to be extracted for each topic. default=10

    Returns:
        Creates and model and writes to model_output_path        
    """
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    model = guidedlda.GuidedLDA(n_topics=n_topics, n_iter=100, random_state=7, refresh=20)
    
    seed_topics = {}
    for t_id, st in enumerate(seed_topic_list):
        for word in st:
            seed_topics[word2id[word]] = t_id
    
    model.fit(dataset, seed_topics=seed_topics, seed_confidence=0.25)
    
    #List the top words of each topic from the trained model. 
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        top_index = np.argsort(topic_dist)[:-(n_top_words+1):-1]
        topic_words = np.array(vocab)[top_index]
        print('Topic [{}]: {}'.format(TOPIC_INDEX[i], ' '.join(topic_words)))
        
    print('\n')
    #Test on some files
    #doc_topic = model.transform(dataset)
    #for i in range(9):
    #    print("Top topic: {} [Document Key words: '{}']".format(TOPIC_INDEX[doc_topic[i].argmax()],
    #                                                  ', '.join(np.array(vocab)[list(reversed(dataset[i,:].argsort()))[0:5]])))
    
    # Dump the model for future production use.
    #model.purge_extra_matrices()
    with open('{}.pickle'.format(model_output_path), 'wb') as file_handle:
        pickle.dump(model, file_handle)


#### Load and convert new data file to DTM
def create_LDAC_file(doc, vocab, output):
    """Function takes a new document of posts and converts to LDA-C format.
        Creates a new file at path `output` which represents the feeds in LDA-C format
        Utility function to create custom datasets.
        
    Args:
        doc (str): Path to the posts file. Each new line represents new post. 
        vocab (list(str)): Global vocab.
        output (str): Output file path.

    Returns:
        na:
    """
    f = open(output, "w")
    rev_word_index = dict(enumerate(vocab, start=1))
    word_index = {v: k for k, v in rev_word_index.items()}

    with open(doc) as fp:  
       line = fp.readline()
       cnt = 1       
       while line:
           #print("Line {}: {}".format(cnt, line.strip()))
           ldac = get_indexed_as_LDAC(line.strip(), word_index)
           if ldac != '0':
               f.write(ldac + '\n')
           else:
               # Write a default word, so that the 
               # line is not empty, and the index is maintained
               f.write('1 17:1\n')
               
           line = fp.readline()
           cnt += 1

    f.close()

def get_indexed_as_LDAC(line, word_index):
    """Utility funtion that converts a string line into LDAC format using the word_index.
        
    Args:
        line (str): String to be converted. 
        word_index (dict): Map of word to the index.

    Returns:
        line in LDAC format
    """
    words = line.split()
    lda_c = ""
    
    #To count unique words in new doc which are also in vocab (We filter out new words)
    count = 0
    
    uniqWords = sorted(set(words)) #remove duplicate words and sort
    for word in uniqWords:
        if word in word_index:
            lda_c =  "{} {}:{}".format(lda_c, word_index[word], words.count(word))
            count = count + 1
            
    lda_c =  "{}{}".format(count, lda_c)
    return lda_c


def create_vocab_from_dataset(dataset_path, output_vocab_path):
    """Utility funtion to create a vocab out of a given dataset 
    
    Args:
        dataset_path (str): Path to the data set
        output_vocab_path (str): Path to the output vocab 
        
    """
    vocab = set()
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    
    dictionary = enchant.Dict("en_US")
    with open(dataset_path) as fp:  
       line = fp.readline()
       while line:
           words = line.strip().split()
           words = [word for word in words if dictionary.check(word)]
           words = [word for word in words if not word in stop_words]    
           vocab.update(words)
           line = fp.readline()

    output_f = open(output_vocab_path, "a")
    for word in vocab:
        output_f.write(word + "\n")

    output_f.close()


def load_vocab(vocab_path):
    """Utility funtion to load a vocab file and create a vocab in GuidedLDA format. 
    
    Args:
        vocab_path (str): Path to the vocab file.
        
    """
    vocab_fn = os.path.join(vocab_path + '.tokens')
    with open(vocab_fn) as f:
        vocab = tuple(f.read().split())
    return vocab


###### Twitter Feed Collection
def clean_post(text, stop_words):
    tokens = word_tokenize(text)

    # convert to lower case
    tokens = [w.lower() for w in tokens]

    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # filter out stop words
    words = [w for w in words if not w in stop_words]
    
    # stemming of words
    #porter = PorterStemmer()
    #stemmed = [porter.stem(word) for word in words]
    return " ".join(str(x) for x in words)


def download_twitter_feed(raw_id_post_output_path, cleaned_posts_output_path, dummy_download=False, count=10):
    """Function fetches most recent twitter messages from the timeline.
        Creates one or two files with recent twitter feeds
        cleaned_posts file is written with cleaned_post_string [```ID is dropped```]
        Sequence number of the posts in the file should be used to map the topic mined posts to original posts.
    
    Args:
        count (int): Count of feeds to be fetched. Max of 200 as supported by twitter.
        raw_id_post_output_path (str): Path where the raw posts with ID file would 
            be downloaded. ```One Post per line``. Line format is ```tweet_id\tpost_string```
        cleaned_posts_output_path (str): Path for the cleaned post file.
        dummy_download (bool): To skip twitter API call. ```raw_id_post_output_path 
            must be a non-empty file with one post per line```

    Returns:
        na
    """
    stop_words = stopwords.words('english')
    cleaned_post_file = open(cleaned_posts_output_path, "w")          
    
    print("CONSUMER_KEY={}".format(CONSUMER_KEY))
    
    if not dummy_download:
        api = twitter.Api(consumer_key=CONSUMER_KEY,
                          consumer_secret=CONSUMER_SECRET,
                          access_token_key=ACCESS_TOKEN_KEY,
                          access_token_secret=ACCESS_TOKEN_SECRET)
        
        statuses = api.GetHomeTimeline(count=count)
        raw_file = open(raw_id_post_output_path, "w")

        for status in statuses:
            post_string = "{} {}" \
                .format(status.text, " ".join(str(x.text) for x in status.hashtags)) \
                .replace('\n', ' ') \
                .replace('\r', '')
            print((status.id, post_string))
            
            raw_file.write("{}\t{}\n".format(status.id, post_string))
            cleaned_post_file.write("{}\n".format(clean_post(post_string, stop_words)))
        
        raw_file.close()
    else:        
        #Clean the file now. 
        cleaned_post_file = open(cleaned_posts_output_path, "a")  
        
        with open(raw_id_post_output_path) as fp:  
           line = fp.readline()
           while line:
               tokens = re.split(r'\t+', line.strip())
               cleaned_post_file.write("{}\n".format(clean_post(tokens[1], stop_words)))
               line = fp.readline()
            
    cleaned_post_file.close()

def get_twitter_feed_by_hashtags(hashtags, cleaned_posts_output_path, count=5):
    """Function fetches most recent twitter messages for a given set of hashtags.
       This is needed to build a good and relevant dataset
    
    Args:
        cleaned_posts_output_path (str): Path for the cleaned post file.
        count (int): Count of feeds to be fetched. Max of 200 as supported by twitter.
        
    Returns:
        na
    """
    stop_words = stopwords.words('english')
    cleaned_post_file = open(cleaned_posts_output_path, "a")          
    
    api = twitter.Api(consumer_key=CONSUMER_KEY,
                      consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_SECRET)

    for hashtag in hashtags:
        print('Getting posts for #{}'.format(hashtag))
        statuses = api.GetSearch(raw_query="q={}&result_type=recent&count={}".format(hashtag, count))
        
        for status in statuses:            
            post_string = "{} {}" \
                .format(status.text, " ".join(str(x.text) for x in status.hashtags)) \
                .replace('\n', ' ') \
                .replace('\r', '')
            #print((status.id, post_string))        
            cleaned_post_file.write("{}\n".format(clean_post(post_string, stop_words)))

        #wait for 15 seconds.
        time.sleep(15)    
    cleaned_post_file.close()


##### Main methods

"""
" vocab_path = "/Users/vkumar/MCS/CS410/Project/GuidedLDA/guidedlda/tests/nyt"
" new_fb_posts_file = "/Users/vkumar/MCS/CS410/Project/data/posts.txt"
" home = "/tmp/
"""
def load_new_data_and_get_topics(raw_posts_file, new_posts_file, vocab, model_path):
    """Function to load a data set and extract topics out of it. 
    
    Args:
        raw_posts_file (str): Path for the raw twitter feed file.
        new_posts_file (str): Path to the cleaned new post feed file. (Same as raw file, but the input is cleaned)
        vocab (vocab): path to vocab path
        model_path (str): Path to the model path. 

    Returns:
        na
    """
    new_data_ldac_file = new_posts_file.replace(".txt", ".ldac")
    create_LDAC_file(new_posts_file, vocab, new_data_ldac_file)
            
    dtm = guidedlda.utils.ldac2dtm(open(new_data_ldac_file), offset=0)
    
    #For testing
    print(dtm.shape)
    print(dtm.sum())
    
    # load the model for prediction
    with open('{}.pickle'.format(model_path), 'rb') as file_handle:
        model = pickle.load(file_handle)

    raw_posts = []
    with open(raw_posts_file) as fp:  
       line = fp.readline()
       while line:
           raw_posts.append(line.strip())
           line = fp.readline()
    
    doc_topic = model.transform(dtm)
    for i in range(len(doc_topic)):
        print("Top topic: {} [Post: '{}']".format(TOPIC_INDEX[doc_topic[i].argmax()], raw_posts[i]))

def train_model_helper(ldac_path, custom_vocab, model_name="guided_lda", defaults=True):
    """Function to train a model using the data in ldac format and custom vocab for the same sata set. 
    
    Args:
        ldac_path (str): Path for the raw twitter feed file.
        custom_vocab (str): Path to the cleaned new post feed file. (Same as raw file, but the input is cleaned)
        model_name (str): Name of the model to be used (as created in training step)
        defaults (bool): Use the default topics or not.

    Returns:
        na
    """
    
    if defaults:
        X = guidedlda.datasets.load_data(guidedlda.datasets.NYT)
        vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)
    else:
        X = guidedlda.utils.ldac2dtm(open(ldac_path), offset=0)
        vocab = custom_vocab

    # Guided LDA with seed topics.
    #Topics 0. Sports 1. Business 2. Arts/Litrature  3.Politics 4. Technology
    seed_topic_list = [['game', 'sport', 'soccer', 'football', 'team', 'win', 'player', 'season', 'second', 'victory'],
                       ['percent', 'company', 'market', 'price', 'sell', 'business', 'stock', 'share'],
                       ['music', 'write', 'art', 'book', 'world', 'film'],
                       ['political', 'government', 'leader', 'official', 'state', 'country', 'american', 'case', 'law', 'police', 'charge', 'officer', 'kill', 'arrest', 'lawyer']]
    
    n_top_words = 10
    n_topics = 5
    model_output_path = HOME + model_name
    train_model(X, vocab, seed_topic_list, model_output_path, n_topics, n_top_words)

def get_twitter_data_and_get_model(tags, output_file, sleep=10):
    """Function which takes a set of hastags and download large amount of twitter feeds for each of these hashtags.
       This is used to create a good dataset. 
       NOTE: Twitter API throttles on developer/basic account. So use must add sleep in between each call.
    
    Args:
        tags (list(str)): List of hashtags to for which the data needs to be downloaded.
        output_file (str): File where these posts need to get written after download.
        sleep (int): Time in seconds between each successive call to twitter.

    Returns:
        Gets a model fitted to the hashtag's posts downloaded.
    """    
    #tags = ('yoga', 'fitness', 'gym', 'crossfit')
    get_twitter_feed_by_hashtags(tags, HOME + output_file, 500) #max 500, twitter does not allow more posts that that    
    
    #Create new vocab by appending new vocab and nyt vocab
    create_vocab_from_dataset(HOME + output_file, HOME + 'tw_posts_all_dump.tokens')
    vocab = guidedlda.datasets.load_vocab(HOME + 'tw_posts_all_dump')
    create_LDAC_file(HOME + output_file, vocab, HOME + 'tw_posts_all_dump.ldac')
    train_model_helper(HOME + 'tw_posts_all_dump.ldac', vocab, model_name="guided_lda", defaults=False)

#### Run the magic!!
def main(argv):
    mode = ''
    feed_input_file = ''
    
    #check default home is set.     
    try:
        opts, args = getopt.getopt(argv, "hm:i:", ["mode=","inputfile="])
    except getopt.GetoptError:
        print('SocialTopicExtractor.py -m <mode[Train|Pull|TestOnline|TestOffline]> -i <feedfile_path>')
        sys.exit(2)
   
    for opt, arg in opts:
        if opt == '-h':
            print('SocialTopicExtractor.py -m <mode[Train|Pull|TestOnline|TestOffline]> -i <feedfile_path>')
            sys.exit()
        elif opt in ("-m", "--mode"):
            mode = arg
        elif opt in ("-i", "--inputfile"):  #Only required for TestOffline mode.
            feed_input_file = arg
         
    vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)

    if mode == 'Train':
        #1. Train default model
        print("Preparing a model by fitting NYT dataset from guidedLDA")
        train_model_helper('', '', 'guided_lda', True)
        vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)
    elif mode == 'Pull':
        #2. Pull twitter feed and fit a model with it. 
        print("Pulling Twitter data for the following hashtags") #Should be put to parameters
        get_twitter_data_and_get_model(('yoga', 'fitness', 'gym', 'crossfit'), "tw_all_hashtag_dump.txt")
    elif mode == 'TestOnline':
        #3. All together: Get latest 20 twitter feed and annotate with topics. 
        print("Getting 20 latest posts from Twitter") 
        download_twitter_feed(count=20, \
                              raw_id_post_output_path = HOME + 'tw_posts_1.txt', \
                              cleaned_posts_output_path = HOME + 'tw_posts_cleaned_1.txt', \
                              dummy_download=False)
        print("Extracting topics for these feeds")
        load_new_data_and_get_topics(HOME + 'tw_posts_1.txt', \
                                     HOME + 'tw_posts_cleaned_1.txt', \
                                     vocab, \
                                     HOME + 'guided_lda')
    elif mode == 'TestOffline':
        #5. Take downloaded twitter feed and run it against a model to get topics for each post.
        print("Cleaning up the input feed data file")
        download_twitter_feed(HOME + feed_input_file, \
                              HOME + 'tw_latest_cleaned.txt', \
                              dummy_download=True)
        print("Extracting topics for these feeds")
        load_new_data_and_get_topics(HOME + 'tw_latest_raw.txt', \
                                     HOME + 'tw_latest_cleaned.txt', \
                                     vocab, HOME + 'guided_lda')

if __name__ == "__main__":
   main(sys.argv[1:])