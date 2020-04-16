# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:56:52 2020

@author: 75109
"""

# Data preprocessing 

import pandas as pd
print('Start to read data.')
df_train = pd.read_csv('ccf_offline_stage1_train.csv')
df_test = pd.read_csv('ccf_offline_stage1_test_revised.csv')
    
# fill the na

df_train=df_train.fillna('null')
df_test=df_test.fillna('null')

print('The number of people with coupon and consumption: ',
          df_train[(df_train['Date_received'] != 'null') & (df_train['Date'] != 'null')].shape[0])
print('The number of people with coupon and without consumption: ',
          df_train[(df_train['Date_received'] != 'null') & (df_train['Date'] == 'null')].shape[0])
print('The number of people without coupon and with consumption: ',
          df_train[(df_train['Date_received'] == 'null') & (df_train['Date'] != 'null')].shape[0])
print('The number of people without coupon and without consumption: ',
          df_train[(df_train['Date_received'] == 'null') & (df_train['Date'] == 'null')].shape[0])
    
    # We need to give coupon to those with consumption and without coupon. 
    # The amount of these people is 701602.

# Label the samples

# y = -1 : means the person did not receive a coupon
# y = 1  : means the person received the coupon and used it within 15 days (positive sample)                 
# y = 0 :  means the person received the coupon but didn't use it within 15 days(negative sample)

print('Label starting: ')
def label(row):
        if row['Date_received'] == 'null':
            return -1
        if row['Date'] != 'null':
            date_buy = pd.to_datetime(row['Date'],format='%Y%m%d')
            date_receive = pd.to_datetime(row['Date_received'],format='%Y%m%d')
            td =  date_buy - date_receive
            if td.days <= 15:
                return 1
        return 0
    
df_train['label'] = df_train.apply(label,axis=1)
df_train['label'].value_counts()
print('Label finished.')
 # We have 988887 negative samples and 64395 positive samples.
