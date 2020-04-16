# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:10:47 2020

@author: 75109
"""

# Prediction of test data

from Model import *
feature_test = ['discount_type','discount_rate','discount_threshold','discount_minus','distance', 'weekday','weekday_type',
                'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'weekday_7','u_coupon_count', 'u_buy_count',
                 'u_buy_with_coupon', 'u_merchant_count', 'u_min_distance', 'u_max_distance', 'u_mean_distance', 'u_median_distance',
                'u_use_coupon_rate', 'u_buy_with_coupon_rate',
             'm_coupon_count', 'm_sale_count', 'm_sale_with_coupon', 'm_min_distance', 'm_max_distance', 'm_mean_distance',
           'm_median_distance', 'm_coupon_use_rate', 'm_sale_with_coupon_rate',
             'um_count', 'um_buy_count', 'um_coupon_count', 'um_buy_with_coupon', 'um_buy_rate', 'um_coupon_use_rate',
           'um_buy_with_coupon_rate']
X_test = test[feature_test]
X_test.head()

df_test_1 = test[['User_id','Coupon_id','Date_received']].copy()
def y_test(method,X_test,df_test_1):
    df_test_1['Probability'] = method.predict_proba(X_test)[:,1]
    return df_test_1
print('Out put the prediction data : ')
print('  LR prediciton examples : ',y_test(pipe_lr,X_test,df_test_1).head())
y_test(pipe_lr,X_test,df_test_1).to_csv('submit_lr.csv',index=False,header=False)
print('  Tree prediciton examples : ',y_test(pipe_tree,X_test,df_test_1).head())
y_test(pipe_tree,X_test,df_test_1).to_csv('submit_tree.csv',index=False,header=False)
print('  Sgdc prediciton examples : ',y_test(pipe_sgdc,X_test,df_test_1).head())
y_test(pipe_sgdc,X_test,df_test_1).to_csv('submit_sgdc.csv',index=False,header=False)
print('   Lgb prediciton examples : ',y_test(pipe_lgb,X_test,df_test_1).head())
y_test(pipe_lgb,X_test,df_test_1).to_csv('submit_lgb.csv',index=False,header=False)
print('ALl the prediction data outputed , please check the files : submit_lr.csv,submit_tree.csv,submit_sgdc.csv & submit_lgb.csv')