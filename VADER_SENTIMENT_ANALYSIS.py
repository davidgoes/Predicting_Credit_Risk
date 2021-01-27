#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediciting Credit Risk with Open Text Data
@author: DavidGoes(6767512)
"""

# =============================================================================
# Sentiment Analysis - dictionary-based - VADER 
# =============================================================================

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
pd.options.mode.chained_assignment = None



df_tweets_all = pd.read_csv('/Users/DavidGoes/Dropbox/CDS prediction/Text data/TWEETS_ALL.csv').drop('Unnamed: 0', axis=1)

print('Tweets loaded')

#Drop empty Tweets
df_tweet_all_dropped = df_tweets_all.dropna(subset=['Tweet'])
df_tweets_all_shape = df_tweets_all.shape #6.1 Mio Tweets
#df_tweet_all_dropped.shape

#Dropped 2085 Tweets at Ticker: CAT
count_dropped_tweets_1 = df_tweets_all_shape[0] - df_tweet_all_dropped.shape[0] 

#Filter Tweets with multiple hastags
df_tweet_all_dropped.loc[:, 'Cashtag'] = df_tweet_all_dropped.Tweet.apply(lambda x: re.findall('\$[^\s]+', x))
df_tweet_all_dropped.loc[:, 'Cashtag_len'] = df_tweet_all_dropped.Cashtag.apply(lambda x: len(x))

#set number of unique Tweets ~20% of Tweets contain more than 5 Cashtags
#df_tweet_all_dropped.Cashtag_len.sort_values().reset_index(drop=True).plot()

#Filter 20% of irrelevant Tweets, remaining: 4.688.534 
df_tweet_all_dropped = df_tweet_all_dropped[df_tweet_all_dropped.Cashtag_len<=5].reset_index(drop=True)

#New data set
df_tweets_all = df_tweet_all_dropped



#pre-processing
#Stop words
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()


#Preprocess Tweets
def normalizer(tweet):
    
    #removing username, cahsticker and hashtag
    only_letters = re.sub('@[^\s]+', '', tweet) #removing username
    only_letters = re.sub('\$[^\s]+', '', only_letters) #removing cashticker
    only_letters = re.sub(r'#([^\s]+)', r'\1', only_letters) #removing #, leaving the word
    
    #removing links
    only_letters = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', only_letters)
    only_letters = re.sub(r'http\S+', '', only_letters)
    
    #removing remaining special signs and number; dont remove things before url 
    only_letters= re.sub('[0-9]', '', only_letters) 
    only_letters= re.sub('[$_@.&+#?:;.,\'\"\-%!*\(\)]', '', only_letters) 

    #lowercase, remove stopwords
    lower_case_letters = only_letters.lower()
    tokens = nltk.word_tokenize(lower_case_letters)
    filtered_result = list(filter(lambda l: l not in stop_words, tokens))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    
    #join tokens for VADER
    tweet = ', '.join(lemmas)
    
    return tweet #only_letters#lower_case_letters #only_letters filtered_result

#Pre-processing Tweets
df_tweets_all.loc[:, 'Tweets_clean'] = df_tweets_all.Tweet.apply(lambda x: normalizer(x))

#Clean data frame
df_tweets_all = df_tweets_all [df_tweets_all .Tweets_clean.isna() != True]\
    .reset_index(drop=True)
df_tweets_all['ID'] = df_tweets_all.index

print('Tweets preprocessed, start sentiment analysis')


#VADER Sentiment analyser
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    #Uncomment following statement display scores
    #print("{:-<40} {}".format(sentence, str(score)))
    
    return score


#Lexical-based sentiment analysis of Tweets
def tweet_sentiment(df_tweets):
 
    tweets = df_tweets['Tweet']
    tweets_cleaned = df_tweets['Tweets_clean']
    df = []
    for i in range(len(tweets)):
        try:
            
            #Uncomment following statement display scores
            #print('Sentiment of tweet: ' + str(i))
            score_1 = sentiment_analyzer_scores(tweets[i])
            score_2 = sentiment_analyzer_scores(tweets_cleaned[i])
            
            #Split sentiment in polarity
            neg_1 = score_1['neg']
            neu_1 = score_1['neu']
            pos_1 = score_1['pos']
            comp_1 = score_1['compound']
            
            neg_2 = score_2['neg']
            neu_2 = score_2['neu']
            pos_2 = score_2['pos']
            comp_2 = score_2['compound'] 
            
            df.append([df_tweets.ID[i], tweets[i], neg_1, neu_1, pos_1, comp_1,\
                       neg_2, neu_2, pos_2, comp_2])
        
        except:
            print('An exception occured at: ID', i)
        
        
    df = pd.DataFrame(df, columns=['ID', 'Tweet', 'neg_1', 'neu_1', 'pos_1', 'comp_1',\
                                   'neg_2', 'neu_2', 'pos_2', 'comp_2']) 
        
    
    return df


#VADER sentiment for all Tweets (pre-processed and social media format)
sent_pre_processed_all = tweet_sentiment(df_tweets_all)

#Filtering of Tweets which contain no  english text. Total of 2085 at ID 2169970
count_dropped_tweets_2 = df_tweets_all.shape[0] - sent_pre_processed_all.shape[0]


df_tweets_all_merged = pd.merge(sent_pre_processed_all, df_tweets_all, \
                                how = 'left', on='ID')
    
df_tweets_all_final = df_tweets_all_merged.drop(axis=1, columns=[\
                                'ID', 'Tweet_x', 'Tweet_y', 'neg', 'neu', 'pos'])
    
df_tweets_all_final = df_tweets_all_final.rename(columns={'Tweets_clean': 'Tweets'})
df_tweets_all_final = df_tweets_all_final[df_tweets_all_final.Tweets.isna() != True].reset_index(drop=True)

#Save final data frame as csv
path  = '/Users/DavidGoes/Desktop/Tweets/TWEETS_ALL_VADER_SENTIMENT.csv' # set path indiviually
#df_tweets_all_final.to_csv(path)

print('New data set has been stored in: ', path)


