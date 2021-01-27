#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediciting Credit Risk with Open Text Data
@author: DavidGoes(6767512)

1. Path to public ticker list CDS_codes_S&P500_companies.xlsx set
2. sart crawler by creating data frame with df = request_splitter(cash_ticker, 20XX)
3. uncomment #df.to_csv and set path to get csv file as output

Note: Time frame is set up to one day to reduce data loss from server errors
Next version will iterate trough days to generate yearly data frames

Process is documented.
"""

import GetOldTweets3 as got
import pandas as pd
import timeit
import time
import calendar

#S&P 500 Tickers publicly stored in github accountR
# =============================================================================
# url_SP = 'https://github.com/davidgoes/Predicting_Credit_Risk/blob/master/CDS_codes_S&P500_companies.xlsx?raw=true'
# df_ticker_sp = pd.read_excel(url_sp)
# =============================================================================

#Ticker list of Dow Jones Industrial Index
url_DJIA = 'https://github.com/davidgoes/Predicting_Credit_Risk/blob/master/ticker_DJII_2010_18.xlsx?raw=true'
df_ticker_DJIA = pd.read_excel(url_DJIA)

ticker_list = df_ticker_DJIA['Ticker']
cash_ticker = '$' + ticker_list
cash_ticker = cash_ticker.astype(str)

#Split list
cash_ticker_10 = cash_ticker[0:10]
cash_ticker_20 = cash_ticker[10:20]
cash_ticker_30 = cash_ticker[20:30]
cash_ticker_37 = cash_ticker[35:]

ticker_dwdp = '$DWDP'
ticker_xom = ['$XOM']

#Year to days
def year_to_list(year):
    year_list = []
    for month in range(0, 12):
        for i in calendar.Calendar().itermonthdates(year, month+1):
           year_list.append(str(i)) 
    
    return year_list #[:3] #uncomment to test crawler for first 3 days


#Tweet crawler
def get_tweets_time_ticker(ticker, since, until): 
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(ticker)\
                                               .setSince(since)\
                                               .setUntil(until)\
                                               
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)
     
    #Create dataframe with metadata  
    df_tweet = []                                   
    for i in range( len(tweet)):
        if len(tweet) != 0:
            df_tweet.append([ticker, tweet[i].date, tweet[i].text, tweet[i].username, 
                             tweet[i].favorites, tweet[i].retweets, tweet[i].hashtags])
        
        #No tweets
        else:
            df_tweet.append([ticker, 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'])
 
           
    df_tweet = pd.DataFrame(df_tweet, columns = ["Ticker", "Date", "Tweet", "Username",
                                           "favorites", "retweets", "hashtags"])
    
    return df_tweet


#Exeptions to restart request after server downtimes
#Exeption in ticker list
def exception(ticker, year_list, sleep):
    print("Restart request in: " + str(sleep))
    
    try:
         tweet = get_tweets_day(ticker, year_list) 
         print("Exception for " + ticker + " solved.")
         
         return tweet
     
    except:
        sleep += 60
        time.sleep(sleep)
        
        return exception(ticker, year_list, sleep)


#Exception in years to days
def exception_2(ticker, since, until, sleep):  
    print("Restart request in: " + str(sleep))
    
    try:
         tweets_day = get_tweets_time_ticker(ticker, since, until)

         return tweets_day
     
    except:
        sleep += 10
        time.sleep(sleep)
        
        return exception_2(ticker, since, until, sleep)
    

#Splitted request for days    
def get_tweets_day(ticker, year_list):
    
    tweets = []
    for i in range(len(year_list)-1):
        since = year_list[i]
        until = year_list[i+1]
        print("Since: " + since + " Until: " + until)
        try:
            tweets_day = get_tweets_time_ticker(ticker, since, until)
        
            print("Append: " + str(len(tweets_day)) + " new tweets.")
            tweets.append(tweets_day)
            
        except:
            print("An exception occured in splitting year to days.")
            
            tweets_day = exception_2(ticker, since, until, 10)
            
            #print("Append: " + str(len(tweets_day)) + " new tweets.")
            
            tweets.append(tweets_day)
            print("Solved exception in splitting year to days.")
            print("Append: " + str(len(tweets_day)) + " new tweets.")
      
    df_tweets = pd.concat(tweets)
    
    return df_tweets


#Split the request of a full query of all S&P 500 tickers throughout a year
def tweets_crawler(ticker_list, year): 
    
    start = timeit.timeit()
    
    year_list = year_to_list(year)
    df_tweets_ticker = [] #data of tweets
    
    for ticker in ticker_list:
        
        try:
            print("Start to crawl tweets for: " + ticker)
            tweet = get_tweets_day(ticker, year_list)
            print(ticker + " has been processed")
            df_tweets_ticker.append(tweet)
                    
            time.sleep(5)
            
            
        except:
            print("An error occured at loc: " + ticker)
            
            time.sleep(10)
            tweet = exception(ticker, year_list, 10)
            df_tweets_ticker.append(tweet)
     
    df_tweets_ticker = pd.concat(df_tweets_ticker, ignore_index=True)
   
    elapsed = timeit.timeit()
    elapsed = elapsed - start
    print("Time spent is: ", elapsed)
            
    return df_tweets_ticker

def get_all_tweets(ticker):
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(ticker)\
                                               .setSince("2017-01-01")\
                                               .setUntil("2017-01-02")\
    
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)
    
    return tweet


#Test data frame for first 10 Tweets + uncomment year list
# =============================================================================
# cash_ticker_2 = cash_ticker[:10].reset_index(drop=True)
# df_test = tweets_crawler(cash_ticker_2, 2018)
# =============================================================================

#Crawl data frame
#Small batches to reduce server errors and data loss
df_2019_10 = tweets_crawler(cash_ticker_10, 2019)
df_2019_20 = tweets_crawler(cash_ticker_20, 2019)
df_2019_30 = tweets_crawler(cash_ticker_20, 2019)
df_2019_37 = tweets_crawler(cash_ticker_20, 2019)

#Copy code and replace -2019- with years to crawl


#Concat seperate frames
df_frames = [df_2019_10, df_2019_20, df_2019_30, df_2019_37]
df_2019 = pd.concat(df_frames) #full data frame containing all tweets of year 2019

#shape 2019: (1519779, 7)
#...
#shape 2016: (2119978, 7)

#Export extracted text data
#Save data frame as csv file to dropbox
#Uncomment and set own path
#df_2019.to_csv('/Users/DavidGoes/Desktop/Tweets/tweets_2019_djia.csv')