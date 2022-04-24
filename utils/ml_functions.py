#Download stopwords corpus if needed
from nltk import download
download('stopwords')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
stopwords_english = stopwords.words('english')
from collections import Counter
import tensorflow as tf
import matplotlib.pyplot as plt
import json

#Stores stopwords as dictionary so we can get O(1) lookup
#Counter counts the list items as keys and the values are the counts
stopwords_dict = Counter(stopwords.words('english'))

def find_optimal_thresholds(fpr, tpr, thresholds):
    unit_vector = np.array([1,1]) / np.linalg.norm(np.array([1,1]))
    distances_squared = []
    for i in range(len(fpr)):
        point = np.array([fpr[i], tpr[i]])
        distances_squared.append((np.linalg.norm(point)**2)-(unit_vector.dot(point)**2))
    return thresholds[np.argmax(distances_squared)]

#Function to filter out url from tweet
def filter_url(string):
    return string[:string.find('https://')].lower()

def tokenize_tweet(string):
    #remove stopwords and tokenize our tweet
    twt_tokenizer = TweetTokenizer()
    return [word for word in twt_tokenizer.tokenize(string) if word not in stopwords_dict]

#index our vocabulary

#determine the size of our vocabulary
num_unique_tokens = 1
unique_tokens = {}
max_tokenized_tweet_length = 0

def count_unique_words(array):
    global num_unique_tokens
    global unique_tokens
    global max_tokenized_tweet_length
    
    max_tokenized_tweet_length = max(len(array), max_tokenized_tweet_length)
    
    for word in array:
        if word not in unique_tokens:
            unique_tokens[word] = num_unique_tokens
            num_unique_tokens += 1
    #Add in 1 more index for words that are not in our corpus
    unique_tokens['WORD NOT IN CORPUS'] = num_unique_tokens
    num_unique_tokens += 1
            
def index_tokenized_array(array):
    return [unique_tokens.get(word,num_unique_tokens-1) for word in array]

def preprocess(dataframe):
    #precprocess tweets
    dataframe['Tweet'] = dataframe['Tweet'].apply(filter_url)
    dataframe['tokenized_tweet'] = dataframe['Tweet'].apply(tokenize_tweet)
    
    #We are going with 70-30 Training, Validation split (1 == Democrat, 0 == Republican)
    dataframe.Party = dataframe.Party.apply(lambda x: int((x == "Democrat")))
    
    #Build and index our Vocabulary
    dataframe.tokenized_tweet.apply(count_unique_words)
    
    #Index our tweet
    dataframe['indexed_tweet'] = dataframe.tokenized_tweet.apply(index_tokenized_array).apply(np.array)
    
    #Pad tweets
    dataframe['padded_sequences'] = tf.keras.preprocessing.sequence.pad_sequences(dataframe.indexed_tweet, maxlen=max_tokenized_tweet_length, padding='post', value=0).tolist()
    
    #Create Modeling Dataset
    X = pd.DataFrame(dataframe['padded_sequences'].to_list(), columns=["token"+str(i) for i in range(1,max_tokenized_tweet_length+1)])
    y = dataframe.Party

    #split dataset
    xTr, xVal, yTr, yVal = train_test_split(X, y, test_size=0.3)
    
    return xTr, xVal, yTr, yVal

#Only used after training
def preprocess_tweet(tweet):
    global num_unique_tokens
    global unique_tokens
    global max_tokenized_tweet_length
    
    f = open('../../config/config.json')
 
    # returns JSON object as
    # a dictionary
    data = json.load(f)

    num_unique_tokens = data['num_unique_tokens']
    unique_tokens = data['unique_tokens']
    max_tokenized_tweet_length = data['max_tokenized_tweet_length']


    # Closing file
    f.close()
    
    #create dataframe of using our tweet
    dataframe = pd.DataFrame(data={"Tweet":tweet}, index=[0])
    
    #precprocess tweets
    dataframe['Tweet'] = dataframe['Tweet'].apply(filter_url)
    dataframe['tokenized_tweet'] = dataframe['Tweet'].apply(tokenize_tweet)
    
    #Index our tweet
    dataframe['indexed_tweet'] = dataframe.tokenized_tweet.apply(index_tokenized_array).apply(np.array)

    #Pad tweets
    dataframe['padded_sequences'] = tf.keras.preprocessing.sequence.pad_sequences(dataframe.indexed_tweet, maxlen=max_tokenized_tweet_length, padding='post', value=0).tolist()
    
    #Create Modeling Dataset
    X = pd.DataFrame(dataframe['padded_sequences'].to_list(), columns=["token"+str(i) for i in range(1,max_tokenized_tweet_length+1)])
    
    return X
    
    


    