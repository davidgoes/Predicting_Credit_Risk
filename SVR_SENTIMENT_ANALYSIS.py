#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediciting Credit Risk with Open Text Data
@author: DavidGoes(6767512)
"""


# =============================================================================
# Sentiment Analysis - model-based - SVR
# =============================================================================

#Import libraries
#Data transformation and visualisation
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import calendar
from datetime import datetime, timedelta 
import statsmodels.api as sm

#NLP
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#ML models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, max_error, r2_score #accuracy_score, 
#import statsmodels.api as sm

#from mlxtend.plotting import plot_decision_regions


df_tweets_all = pd.read_csv('/Users/DavidGoes/Desktop/Tweets/TWEETS_ALL_VADER_SENTIMENT.csv').drop('Unnamed: 0', axis=1)

#Cleaning (77_681 will be removed) 
df_tweets_all = df_tweets_all [df_tweets_all.Tweets.isna() != True].reset_index(drop=True)
#df_tweets_all = df_tweets_all.rename(columns={'Tweet': 'Tweets'})
    
df_credit_risk = pd.read_csv('/Users/DavidGoes/Desktop/ALL_VARS.csv')

#Merge Text data and Financials
df_tweets = df_tweets_all.drop(axis=1, columns=['Username', 'neg_1', 'neu_1', 'pos_1', 'comp_1', 'neg_2', 'neu_2',\
                               'pos_2', 'comp_2']) #, 'favorites', 'retweets', 'hashtags' #included in df_financials
    
df_VADER_sent = df_tweets_all[['Ticker', 'Date', 'Username', 'neg_1', 'neu_1', 'pos_1', 'comp_1', 'neg_2', 'neu_2',\
                               'pos_2', 'comp_2', 'favorites', 'retweets', 'hashtags']]

#Combine Tweets on daily basiss
df_tweets_grouped = df_tweets.groupby(['Date', 'Ticker'])['Tweets'] #check how VADER sent is grouped
df_tweets_joined = df_tweets_grouped.apply(lambda x: "[%s]" % ', '.join(x)) #join all tweets

#Aggregate VADER sentiment on daily basis
df_VADER_sent_grouped = df_VADER_sent.groupby(['Date', 'Ticker']).agg({'comp_1': 'mean', 'comp_2': 'mean', #Sentiment scores
                                                              'neg_1': 'mean', 'neg_2': 'mean',
                                                              'neu_1': 'mean', 'neu_2': 'mean',
                                                              'pos_1': 'mean', 'pos_2': 'mean',})
                                                              #'Username': 'count', #activity
                                                              #'favorites': 'sum',
                                                              #'retweets': 'sum',  
                                                              #'hashtags': 'count'

#Merge Tweets and sentiment
df_tweets_all_merged = pd.merge(df_tweets_joined, df_VADER_sent_grouped, on=['Date', 'Ticker'])

#Merge data frame for SVR
df_merged_svr = pd.merge(df_tweets_all_merged, df_credit_risk , on=['Date', 'Ticker'])

#Add variables#Included in df merged
#df_merged_svr['datetime'] =  pd.to_datetime(df_merged_svr.Date, infer_datetime_format=True)
#df_merged_svr['Quarter'] = df_merged_svr.datetime.dt.quarter


#Support Vector Regression
print('Starting SVR...')
#Term Document Matrix
count_vectorizer = CountVectorizer(ngram_range=(1,3), max_features=7500) #unigrams, bigrams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(3,3), max_features=7500)

#Test
#df_merged_test = df_merged_svr

#df_train = df_merged_test[df_merged_test.Year==2016]
#df_test = df_merged_test[df_merged_test.Year!=2016]


#For each firm individually
def svr_results_all(df_all):
    
    #df_merged_svr_new_vars= df_merged_svr_new_vars.set_index(df_merged_svr_new_vars.datetime)
    df_collection = []
    for Ticker in df_all.Ticker.unique():
        
        print('Start with Ticker: ', Ticker)
        df = df_all[df_all.Ticker == Ticker]
        df = df.sort_values(by=['Date'])
        df = df.set_index(df.datetime)
        
        try:
            
            print('Start:', Ticker)
            df = df.dropna(axis=0, subset=['CDS_pct_change']) #first value of pct_change is NaN 
            
            df_train = df[df.Year==2016]
            df_test = df[df.Year!=2016]

            Train_X = df_train.Tweets #.reset_index(drop=True)
            Test_X = df_test.Tweets #.reset_index(drop=True)

            Train_Y = df_train.CDS #.reset_index(drop=True)
            Test_Y = df_test.CDS #.reset_index(drop=True)
            
            Train_pct_change_Y = df_train.CDS_pct_change #.reset_index(drop=True)
            Test_pct_change_Y = df_test.CDS_pct_change #.reset_index(drop=True)
            
            print('Clear_1')
            
            print('Start: TDM')

            Train_X_vectorized = tfidf_vectorizer.fit_transform(Train_X).astype(float)
            Test_X_vectorized = tfidf_vectorizer.fit_transform(Test_X).astype(float)

            print('Split compleeted, start model creation...')

            model = svm.SVR(kernel='rbf', C=1, degree=7, epsilon=0.04, gamma=1)
            model.fit(Train_X_vectorized, Train_Y)
            y_pred_cds = model.predict(Test_X_vectorized)
            
            model_pct_change = svm.SVR(kernel='rbf', C=1, degree=7, epsilon=0.04, gamma=1)
            model_pct_change.fit(Train_X_vectorized, Train_pct_change_Y)
            y_pred_pct_change = model_pct_change.predict(Test_X_vectorized)

            score = model.score(Train_X_vectorized, Train_Y)
            mae = mean_absolute_error(Test_Y, y_pred_cds)

            exp_var= explained_variance_score(Test_Y, y_pred_cds)
            Max_error = max_error(Test_Y, y_pred_cds) 
            r2 = r2_score(Test_Y, y_pred_cds)
            
            plt.scatter(y_pred_cds, Test_Y)
            plt.show()

            print("Model ", score, " Test: ", r2, "Mae ", mae, "Max error ", Max_error, "Explained variance ", exp_var)
            

            #results.append([Ticker, y_pred_cds, Test_Y, score, r2, mae])
            
            print('Data frame of predictions (abolute values and returns).')
            
            df_pred = {'Pred_CDS': y_pred_cds, 'Pred_CDS_pct_change': y_pred_pct_change}
            
            predictions = pd.DataFrame(df_pred, index=Test_Y.index)
            
            #print(pred.columns)
            print(predictions.shape)

            #Merge predicted values on timestamp
            df_merged = pd.merge(df, predictions, how = 'left',left_index = True, right_index = True)
            print(df_merged.shape)
            #print(df_merged.columns)
            
            #df_collection.append(y_pred_cds)
            
            #prediction = pd.DataFrame(y_pred_svr, colums=['Prediction'])
            
            #df.append(prediction)
            
            df_collection.append(df_merged)
            
            print('Something has been appended')
        
        except:
            print('An exception occured at: ', Ticker)
            
    df_new = pd.concat(df_collection)
        
    return df_new

df_svr = svr_results_all(df_merged_svr)


#For each firm individually
def svr_results_all_2(df_all):
    
    #df_merged_svr_new_vars= df_merged_svr_new_vars.set_index(df_merged_svr_new_vars.datetime)
    df_results = []
    df_params = []
    #Hyperparameter selection
    #Params
    Cs = [1, 3, 10, 50, 100] #30, 40, 50, 80, 90
    gammas = [0.01, 0.1, 1] #2
    degrees = [1, 3, 5, 7, 10] # 1, 2, 4, 5, 6, 7, 8, 9, 10
    epsilons = [0.001, 0.002, 0.004, 1,0] #,0.6, 0.8, 1,0, 1.2, 1.4 , 1.5
    kernel_list = ['sigmoid', 'rbf', 'poly']
    param_grid = {'C': Cs, 'gamma' : gammas, 'degree' : degrees, \
                  'epsilon' : epsilons, 'kernel': kernel_list}
    
    for Ticker in df_all.Ticker.unique():
        
        print('Start with Ticker: ', Ticker)
        df = df_all[df_all.Ticker == Ticker]
        df = df.sort_values(by=['Date'])
        df = df.set_index(df.datetime)
        
        try:
            
            print('Start:', Ticker)
            df = df.dropna(axis=0, subset=['CDS_pct_change']) #first value of pct_change is NaN 
            
            df_train = df[df.Year==2016]
            df_test = df[df.Year!=2016]

            Train_X = df_train.Tweets #.reset_index(drop=True)
            Test_X = df_test.Tweets #.reset_index(drop=True)

            Train_Y = df_train.CDS #.reset_index(drop=True)
            Test_Y = df_test.CDS #.reset_index(drop=True)
            
            Train_pct_change_Y = df_train.CDS_pct_change #.reset_index(drop=True)
            Test_pct_change_Y = df_test.CDS_pct_change #.reset_index(drop=True)
            
            print('Clear_1')
            
            print('Start: TDM')

            Train_X_vectorized = tfidf_vectorizer.fit_transform(Train_X).astype(float)
            Test_X_vectorized = tfidf_vectorizer.fit_transform(Test_X).astype(float)
            
            #Model 1
            print('MODEL_1_CDS_absolute: Hyperparameter selection.')
            
            grid_search = GridSearchCV(svm.SVR(), param_grid=param_grid)
            grid_search.fit(Train_X_vectorized, Train_Y)
            param = grid_search.best_params_
            df_param = pd.DataFrame.from_dict(param, orient='index').T
            
            #print('Split compleeted, start model creation...')
            model = svm.SVR(kernel=param['kernel'], C=param['C'], 
                            degree=param['degree'], epsilon=param['epsilon'], 
                            gamma=param['gamma'])
            
            model.fit(Train_X_vectorized, Train_Y) #first Train variable
            y_pred_cds = model.predict(Test_X_vectorized)
            
            #Model performance
            score = model.score(Train_X_vectorized, Train_Y)
            mae = mean_absolute_error(Test_Y, y_pred_cds)
            mse = mean_squared_error(Test_Y, y_pred_cds)
            exp_var= explained_variance_score(Test_Y, y_pred_cds)
            Max_error = max_error(Test_Y, y_pred_cds) 
            r2 = r2_score(Test_Y, y_pred_cds)
            
            print("Model: ", score, " R_squared: ", r2, "Mae: ", mae, "Max error: ", "MSE: ", mse
                  , Max_error, "Explained variance: ", exp_var,
                  'Parameter selection: ', param)
            
            plt.scatter(y_pred_cds, Test_Y)
            plt.show()
            
            #Store values in data frame for model assessement
            dict_performance_1 = {'Score_1': score, 'r_squared_1': r2, 'MAE_1': mae,
                                'MSE_1': mse, 'Max_error_1': Max_error, 
                                'exp_var_1': exp_var}
            
            print('create data frame...')
            #df_param = df_param_pct_change.append(dict_performance, ignore_index=True)
            df_performance_1 = pd.DataFrame.from_dict(dict_performance_1, orient='index').T
            print('dict')
            df_svr_param_perf_1 = pd.concat([df_param, df_performance_1], axis=1)
            print('concat')
            df_svr_param_perf_1['Ticker'] = Ticker
            #df_params.append([df_param, df_performance_1])
            df_params.append(df_svr_param_perf_1)
            
            
            #Model 2
            print('MODEL_2_CDS_pct_change: Hyperparameter selectio.')
            
            grid_search_pct_change = GridSearchCV(svm.SVR(), param_grid=param_grid)
            grid_search_pct_change.fit(Train_X_vectorized, Train_pct_change_Y)
            param_pct_change = grid_search_pct_change.best_params_
            df_param_pct_change = pd.DataFrame.from_dict(param_pct_change, orient='index').T
            
            print('Split compleeted, start model creation...')
            model_pct_change = svm.SVR(kernel=param_pct_change['kernel'], C=param_pct_change['C'], 
                            degree=param_pct_change['degree'], epsilon=param_pct_change['epsilon'], 
                            gamma=param_pct_change['gamma'])
        
            model_pct_change.fit(Train_X_vectorized, Train_pct_change_Y) #second Train variable
            y_pred_cds_pct_change = model_pct_change.predict(Test_X_vectorized) #predicted values
            

            #Model 2 performance
            score_pct_change = model.score(Train_X_vectorized, Train_Y)
            r2_pct_change = r2_score(Test_pct_change_Y, y_pred_cds_pct_change)
            mae_pct_change = mean_absolute_error(Test_pct_change_Y, y_pred_cds_pct_change)
            mse_pct_change = mean_squared_error(Test_pct_change_Y, y_pred_cds_pct_change)
            Max_error_pct_change = max_error(Test_pct_change_Y, y_pred_cds_pct_change) 
            exp_var_pct_change = explained_variance_score(Test_pct_change_Y, y_pred_cds_pct_change)
            
            plt.scatter(y_pred_cds_pct_change, Test_pct_change_Y)
            plt.show()
            
            print("Model pct_change: ", score_pct_change, "R_squared:  ", r2_pct_change, "MAE: ", mae_pct_change, 
                  "Max error: ", Max_error_pct_change, "MSE: ", mse_pct_change,
                  "Explained variance: ", exp_var_pct_change, 'Parametervselection: ', param)
            
            #Store values in data frame for model assessement
            dict_performance_2 = {'Score_2': score_pct_change, 'r_squared_2': r2_pct_change, 'MAE_2': mae_pct_change,
                                'MSE_2': mse_pct_change, 'Max_error_2': Max_error_pct_change, 
                                'exp_var_2': exp_var_pct_change}
            
            print('create data frame...')
            #df_param_pct_change = df_param_pct_change.append(dict_performance, ignore_index=True)
            df_performance_2 = pd.DataFrame.from_dict(dict_performance_2, orient='index').T
            df_svr_param_perf_2 = pd.concat([df_param_pct_change, df_performance_2], axis=1)
            df_svr_param_perf_2['Ticker'] = Ticker
            #df_params.append([df_param_pct_change, df_performance_2])
            df_params.append(df_svr_param_perf_2)
                    

            #results.append([Ticker, y_pred_cds, Test_Y, score, r2, mae])
            
            print('Pred data frame')
            
            dict_pred = {'Pred_CDS': y_pred_cds, 'Pred_CDS_pct_change': y_pred_cds_pct_change}
            df_pred = pd.DataFrame(dict_pred, index=Test_Y.index)
            
            #print(pred.columns)
            print(df_pred.shape)

            #Merge predicted values on timestamp
            df_merged = pd.merge(df, df_pred, how = 'left',left_index = True, right_index = True)
            print(df_merged.shape)
            #print(df_merged.columns)
            
            #df_collection.append(y_pred_cds)
            
            #prediction = pd.DataFrame(y_pred_svr, colums=['Prediction'])
            
            #df.append(prediction)
            
            df_results.append(df_merged)
            
            print('Something has been appended')
        
        except:
            print('An exception occured at: ', Ticker)
            
    df_final = pd.concat(df_results)
    df_params_final = pd.concat(df_params)
        
    return [df_final, df_params_final]

df_merged_svr_results_all = svr_results_all_2(df_merged_svr)


df_merged_svr_results = df_merged_svr_results_all[0]
df_merged_svr_performance = df_merged_svr_results_all[1]

hyperparamters_mode = df_merged_svr_performance[df_merged_svr_performance.columns[:4]].mode()

# =============================================================================
# df_svr
# Out[2]: {'C': 20, 'degree': 3, 'epsilon': 0.001, 'gamma': 0.01, 'kernel': 'poly'}
# 
# =============================================================================

#Q why so light values
#Visualisation of results
def stats_svr(df_all):
    df_list = []
    for Ticker in df_all.Ticker.unique():
        df = df_all[df_all.Ticker == Ticker]
        #df = df.sort_values(by=['Date'])
        
        print(Ticker)
        
        df.Date = df.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        df.set_index('Date', inplace=True)
        
        #plt.figure(figsize=(20, 8))
        #plt.plot(df.Pred_CDS)
        #plt.plot(df.comp)

        #plt.legend()
        #plt.title('Moving Average (5, 75)')
        #plt.ylabel('Sentiment (SVR Prediciton)');
        #plt.show()
        df['CDS_norm'] = (df.CDS-df.CDS.min())/(df.CDS.max()-df.CDS.min())
        
        plt.figure(figsize=(20, 8))
        #plt.plot(df.CDS_pct_change)
        plt.plot(df.CDS_norm)
        plt.plot(df.Pred_CDS)
        
        #plt.plot(df.PRC)

        plt.legend()
        plt.title('CDS / Stock prices')
        plt.ylabel('CDS spreads / Stock prices');
        plt.show() 
        
        
stats_svr(df_merged_svr_results)


#Save data to csv/excel
df_merged_svr_results_final = df_merged_svr_results.drop(['Tweets', 'Unnamed: 0'], axis=1)

#df_merged_svr_results_final.to_csv('/Users/DavidGoes/Desktop/Sentiment/SVR_SENTIMENT.csv')
#df_merged_svr_performance.to_excel('/Users/DavidGoes/Desktop/SVR Data/SVR_PERFORMANCE.xlsx', index=False)
