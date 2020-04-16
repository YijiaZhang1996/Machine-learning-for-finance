# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:00:50 2020

@author: 75109
"""
from data_preprocessing import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import  auc, roc_curve

# Features construction

# 2.1 Discount rate
print('Extracting the discount rate features : ')
# Transform Discount rate in to more features : 
#        discount_type,discount_rate, discount_threshold,discount_minus 
        
df_train.Discount_rate.unique()

    # whether the discount rate is null
    
def getDiscountType(row):
    if 'null' in row:
        return 0
    elif ':' in row:
        return 1
    else:
        return 2

    #actual discount rate
    
def convertRate(row):
    if 'null' in row:
        return 1
    elif ':' in row:
        money = row.split(':')
        rate = 1.0 - float(money[1])/float(money[0])
        return rate
    else:
        return float(row)
        
    #the threshold of discount
def getDiscount_threshold(row):
    if ':' in row:
        money = row.split(':')
        return int(money[0])
    elif 'null' in row:
        return 'null'
    else:
        return 0
    
#the discount amount
def getDiscount_minus(row):
    if ':' in row:
        money = row.split(':')
        return int(money[1])
    elif 'null' in row:
        return 'null'
    else:
        return 0

    #apply 4 functions
    
def processData(df):
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_threshold'] = df['Discount_rate'].apply(getDiscount_threshold)
    df['discount_minus'] = df['Discount_rate'].apply(getDiscount_minus)
    print('discount rate %s' %df['discount_rate'].unique()) #actual discount rate
    return df

df_train = processData(df_train)
df_test = processData(df_test)

# 2.2 Distance
print('Extracting the distance features : ')
df_train['distance'] = df_train['Distance'].replace('null',-1).astype(int)
df_test['distance'] = df_test['Distance'].replace('null',-1).astype(int)
df_train['distance'] .unique()
df_train.info()

# 2.3 The date of receiving the coupon

# range of date of receiving the coupon
print('Extracting the date features : ')
date_receive = df_train['Date_received'].unique()
date_receive = sorted(date_receive[date_receive != 'null'])
print('The date of receiving the coupon：%d - %d'%(date_receive[0],date_receive[-1]))
    
#range of date of purchase
    
date_buy = df_train['Date'].unique()
date_buy = sorted(date_buy[date_buy != 'null'])
print('The date of purchase：%d - %d'%(date_buy[0],date_buy[-1]))

# plot the figure: the ratio of coupon_used/ coupon_received

couponbydate = df_train[df_train['Date_received'] != 'null'][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
couponbydate.columns = ['Date_received','count']
buybydate = df_train[(df_train['Date'] != 'null') & (df_train['Date_received'] != 'null')][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
buybydate.columns = ['Date_received','count']



sns.set_style('ticks')
sns.set_context("notebook", font_scale= 1.4)
plt.figure(figsize = (12,8))
date_received_dt = pd.to_datetime(date_receive, format='%Y%m%d')

plt.subplot(211)
plt.bar(date_received_dt, couponbydate['count'], label = 'number of coupon received' )
plt.bar(date_received_dt, buybydate['count'], label = 'number of coupon used')
plt.yscale('log')
plt.ylabel('Count')
plt.legend()

plt.subplot(212)
plt.bar(date_received_dt, buybydate['count']/couponbydate['count'])
plt.ylabel('Ratio(coupon used/coupon received)')
plt.tight_layout()


#construct the feature of 'weekday'


def getWeekday(row):
    if row == 'null':
        return row
    else:
        weekday = datetime.date(int(row[0:4]),int(row[4:6]),int(row[6:8])).weekday() + 1
        return weekday
df_train['weekday'] = df_train['Date_received'].astype(str).apply(getWeekday)
df_test['weekday'] = df_test['Date_received'].astype(str).apply(getWeekday)
print(df_train['weekday'].unique())
    
#construct the feature of 'weekday_type'
    
df_train['weekday_type'] = df_train['weekday'].apply(lambda x: 1 if x in [6,7] else 0)
df_test['weekday_type'] = df_test['weekday'].apply(lambda x: 1 if x in [6,7] else 0)
df_train['weekday_type'] .unique()
# If the date is weekday, weekday_type=0. If the date is weekend, weekday_type=1

##construct the feature of 'weekday_number'
#training data
#one-hot-encoding
data = df_train['weekday'].replace('null',np.nan)
tmpdf = pd.get_dummies(data,prefix='weekday')
df_train = pd.concat([df_train,tmpdf],axis=1)

#test data
#one-hot-encoding
data = df_test['weekday'].replace('null',np.nan)
tmpdf = pd.get_dummies(data,prefix='weekday')
df_test = pd.concat([df_test,tmpdf],axis=1)
        
#We got 7 dummy variables representing each day of a week

# 2.4 Features related to Consumers,Coupon_id & mechants
    
# Get UserFeature
def userFeature(df):
    u = df[['User_id']].copy().drop_duplicates()
    
    # u_coupon_count : num of coupon received by user
    u1 = df[df['Date_received'] != 'null'][['User_id']].copy()
    u1['u_coupon_count'] = 1
    u1 = u1.groupby(['User_id'], as_index = False).count()

    # u_buy_count : times of user buy offline (with or without coupon)
    u2 = df[df['Date'] != 'null'][['User_id']].copy()
    u2['u_buy_count'] = 1
    u2 = u2.groupby(['User_id'], as_index = False).count()

    # u_buy_with_coupon : times of user buy offline (with coupon)
    u3 = df[((df['Date'] != 'null') & (df['Date_received'] != 'null'))][['User_id']].copy()
    u3['u_buy_with_coupon'] = 1
    u3 = u3.groupby(['User_id'], as_index = False).count()

    # u_merchant_count : num of merchant user bought from
    u4 = df[df['Date'] != 'null'][['User_id', 'Merchant_id']].copy()
    u4.drop_duplicates(inplace = True)
    u4 = u4.groupby(['User_id'], as_index = False).count()
    u4.rename(columns = {'Merchant_id':'u_merchant_count'}, inplace = True)

    # min,max,mean and median diatance of purchase with coupon
    utmp = df[(df['Date'] != 'null') & (df['Date_received'] != 'null')][['User_id', 'distance']].copy()
    utmp.replace(-1, np.nan, inplace = True)
    u5 = utmp.groupby(['User_id'], as_index = False).min()
    u5.rename(columns = {'distance':'u_min_distance'}, inplace = True)
    u6 = utmp.groupby(['User_id'], as_index = False).max()
    u6.rename(columns = {'distance':'u_max_distance'}, inplace = True)
    u7 = utmp.groupby(['User_id'], as_index = False).mean()
    u7.rename(columns = {'distance':'u_mean_distance'}, inplace = True)
    u8 = utmp.groupby(['User_id'], as_index = False).median()
    u8.rename(columns = {'distance':'u_median_distance'}, inplace = True)

    #merge all the features on User_id
    user_feature = pd.merge(u, u1, on = 'User_id', how = 'left')
    user_feature = pd.merge(user_feature, u2, on = 'User_id', how = 'left')
    user_feature = pd.merge(user_feature, u3, on = 'User_id', how = 'left')
    user_feature = pd.merge(user_feature, u4, on = 'User_id', how = 'left')
    user_feature = pd.merge(user_feature, u5, on = 'User_id', how = 'left')
    user_feature = pd.merge(user_feature, u6, on = 'User_id', how = 'left')
    user_feature = pd.merge(user_feature, u7, on = 'User_id', how = 'left')
    user_feature = pd.merge(user_feature, u8, on = 'User_id', how = 'left')
    
    # calculate rate
    #每个用户的优惠券使用率
    user_feature['u_use_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float')/user_feature['u_coupon_count'].astype('float')
    #每个用户线下消费中用券的比例
    user_feature['u_buy_with_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float')/user_feature['u_buy_count'].astype('float')
    user_feature = user_feature.fillna(0)
    
    print(user_feature.columns.tolist())
    return user_feature

# Get Merchant Featuer 

def merchantFeature(df):
    m = df[['Merchant_id']].copy().drop_duplicates()

    # m_coupon_count : num of coupon from merchant 每个商家发放的优惠券数量
    m1 = df[df['Date_received'] != 'null'][['Merchant_id']].copy()
    m1['m_coupon_count'] = 1
    m1 = m1.groupby(['Merchant_id'], as_index = False).count()

    # m_sale_count : num of sale from merchant (with or without coupon) 每个商家销售次数（用券和不用券）
    m2 = df[df['Date'] != 'null'][['Merchant_id']].copy()
    m2['m_sale_count'] = 1
    m2 = m2.groupby(['Merchant_id'], as_index = False).count()

    # m_sale_with_coupon : num of sale from merchant with coupon usage 每个商家用券的销售次数
    m3 = df[(df['Date'] != 'null') & (df['Date_received'] != 'null')][['Merchant_id']].copy()
    m3['m_sale_with_coupon'] = 1
    m3 = m3.groupby(['Merchant_id'], as_index = False).count()

    # 每个商家最小、最大、平均、中位数的购买距离
    mtmp = df[(df['Date'] != 'null') & (df['Date_received'] != 'null')][['Merchant_id', 'distance']].copy()
    mtmp.replace(-1, np.nan, inplace = True)
    m4 = mtmp.groupby(['Merchant_id'], as_index = False).min()
    m4.rename(columns = {'distance':'m_min_distance'}, inplace = True)
    m5 = mtmp.groupby(['Merchant_id'], as_index = False).max()
    m5.rename(columns = {'distance':'m_max_distance'}, inplace = True)
    m6 = mtmp.groupby(['Merchant_id'], as_index = False).mean()
    m6.rename(columns = {'distance':'m_mean_distance'}, inplace = True)
    m7 = mtmp.groupby(['Merchant_id'], as_index = False).median()
    m7.rename(columns = {'distance':'m_median_distance'}, inplace = True)

    merchant_feature = pd.merge(m, m1, on = 'Merchant_id', how = 'left')
    merchant_feature = pd.merge(merchant_feature, m2, on = 'Merchant_id', how = 'left')
    merchant_feature = pd.merge(merchant_feature, m3, on = 'Merchant_id', how = 'left')
    merchant_feature = pd.merge(merchant_feature, m4, on = 'Merchant_id', how = 'left')
    merchant_feature = pd.merge(merchant_feature, m5, on = 'Merchant_id', how = 'left')
    merchant_feature = pd.merge(merchant_feature, m6, on = 'Merchant_id', how = 'left')
    merchant_feature = pd.merge(merchant_feature, m7, on = 'Merchant_id', how = 'left')

    merchant_feature['m_coupon_use_rate'] = merchant_feature['m_sale_with_coupon'].astype('float')/merchant_feature['m_coupon_count'].astype('float')
    merchant_feature['m_sale_with_coupon_rate'] = merchant_feature['m_sale_with_coupon'].astype('float')/merchant_feature['m_sale_count'].astype('float')
    merchant_feature = merchant_feature.fillna(0)
    
    print(merchant_feature.columns.tolist())
    return merchant_feature

# Get Uer_mechant Feature
def usermerchantFeature(df):
    
    # key of user and merchant 顾客-商家pair的数量
    um = df[['User_id', 'Merchant_id']].copy().drop_duplicates()
    
    #每对pair的出现次数？
    um1 = df[['User_id', 'Merchant_id']].copy()
    um1['um_count'] = 1
    um1 = um1.groupby(['User_id', 'Merchant_id'], as_index = False).count()
   
    #每对pair的交易次数
    um2 = df[df['Date'] != 'null'][['User_id', 'Merchant_id']].copy()
    um2['um_buy_count'] = 1
    um2 = um2.groupby(['User_id', 'Merchant_id'], as_index = False).count()
    
    #每对pair的发券次数
    um3 = df[df['Date_received'] != 'null'][['User_id', 'Merchant_id']].copy()
    um3['um_coupon_count'] = 1
    um3 = um3.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    #每对pair的用券消费次数
    um4 = df[(df['Date_received'] != 'null') & (df['Date'] != 'null')][['User_id', 'Merchant_id']].copy()
    um4['um_buy_with_coupon'] = 1
    um4 = um4.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    user_merchant_feature = pd.merge(um, um1, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um2, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um3, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um4, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = user_merchant_feature.fillna(0)

    #calculate rate
    user_merchant_feature['um_buy_rate'] = user_merchant_feature['um_buy_count'].astype('float')/user_merchant_feature['um_count'].astype('float')
    user_merchant_feature['um_coupon_use_rate'] = user_merchant_feature['um_buy_with_coupon'].astype('float')/user_merchant_feature['um_coupon_count'].astype('float')
    user_merchant_feature['um_buy_with_coupon_rate'] = user_merchant_feature['um_buy_with_coupon'].astype('float')/user_merchant_feature['um_buy_count'].astype('float')
    user_merchant_feature = user_merchant_feature.fillna(0)

    print(user_merchant_feature.columns.tolist())
    return user_merchant_feature

# Extract Features of data
def featureProcess(feature,train, test):
    
    user_feature = userFeature(feature)
    merchant_feature = merchantFeature(feature)
    user_merchant_feature = usermerchantFeature(feature)
    
    feature = pd.merge(feature, user_feature, on = 'User_id', how = 'left')
    feature = pd.merge(feature, merchant_feature, on = 'Merchant_id', how = 'left')
    feature = pd.merge(feature, user_merchant_feature, on = ['User_id', 'Merchant_id'], how = 'left')
    feature = feature.fillna(0)
    
    train = pd.merge(train, user_feature, on = 'User_id', how = 'left')
    train = pd.merge(train, merchant_feature, on = 'Merchant_id', how = 'left')
    train = pd.merge(train, user_merchant_feature, on = ['User_id', 'Merchant_id'], how = 'left')
    train = train.fillna(0)
    
    test = pd.merge(test, user_feature, on = 'User_id', how = 'left')
    test = pd.merge(test, merchant_feature, on = 'Merchant_id', how = 'left')
    test = pd.merge(test, user_merchant_feature, on = ['User_id', 'Merchant_id'], how = 'left')
    test = test.fillna(0)
    
    return feature,train, test


df_train['Date_received'] = df_train['Date_received'].astype(str)
df_train['Date'] = df_train['Date'].astype(str)
df_train['Date_received'].unique()
#用于构造特征
feature = df_train[(df_train['Date'] < '20160516') | ((df_train['Date'] == 'null') & (df_train['Date_received'] < '20160516'))].copy()
#用于训练
data = df_train[(df_train['Date_received'] >= '20160516') & (df_train['Date_received'] <= '20160615')].copy()
print(data['label'].value_counts())

print('Start to Extract Features of data :')
feature,train_valid,test = featureProcess(feature,data,df_test) 
print(feature['label'].value_counts())
print(train_valid['label'].value_counts())
print('Start to split Train and Valid Set :')
train, valid = train_test_split(train_valid, test_size = 0.1, stratify = train_valid['label'], random_state=100)
print('Train Set:\n',train['label'].value_counts())
print('Valid Set:\n',valid['label'].value_counts())


# All the features
print('All the columns:')
for i in df_train.columns:
    print('\t',i)















