# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:25:18 2020

@author: 75109
"""

# LR Model construction
from features_construction import *
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Establish X_train y_train X_valid y_valid
print('Establish X_train y_train X_valid y_valid : ')

feature = ['discount_type','discount_rate','discount_threshold','discount_minus','distance','weekday','weekday_type',
           'weekday_1.0','weekday_2.0','weekday_3.0','weekday_4.0','weekday_5.0','weekday_6.0','weekday_7.0','u_coupon_count', 'u_buy_count',
           'u_buy_with_coupon', 'u_merchant_count', 'u_min_distance', 'u_max_distance', 'u_mean_distance', 'u_median_distance', 'u_use_coupon_rate',
           'u_buy_with_coupon_rate',
            'm_coupon_count', 'm_sale_count', 'm_sale_with_coupon', 'm_min_distance', 'm_max_distance', 'm_mean_distance',
           'm_median_distance', 'm_coupon_use_rate', 'm_sale_with_coupon_rate',
            'um_count', 'um_buy_count', 'um_coupon_count', 'um_buy_with_coupon', 'um_buy_rate', 'um_coupon_use_rate','um_buy_with_coupon_rate']
X_train = train[feature]
X_valid = valid[feature]
y_train = train['label']
y_valid = valid['label']
print('Establish X_train y_train X_valid y_valid Finished. ')

# Defint function AUC_calculate
def get_pre_prob(x,valid,X_valid):
    valid_ = valid.copy()
    valid_['pred_prob'] = x.predict_proba(X_valid)[:,1]
    return valid_
def AUC_calculate(x,valid,X_valid):
    valid_groupby = get_pre_prob(x,valid,X_valid).groupby(['Coupon_id'])
    aucs = []
    mean_tpr = 0.0
    for i in valid_groupby:
        tmpdf = i[1]
        if len(tmpdf['label'].unique())==1:
            continue
        fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
        aucs.append(auc(fpr,tpr))
    return np.mean(aucs)


## Lr
print('Start lr model :')
from sklearn.linear_model import LogisticRegression
pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(random_state=1, penalty = 'l1',solver = 'liblinear'))

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_valid)
print('    Training accuracy of lr:', pipe_lr.score(X_train, y_train))
print('    Valid Accuracy of lr: %.8f' % pipe_lr.score(X_valid, y_valid))
print('    AUC value of lr : ',AUC_calculate(pipe_lr,valid,X_valid))
print('End lr model .')

## Decision Tree
print('Start tree model :')
from sklearn.tree import DecisionTreeClassifier
pipe_tree = make_pipeline(StandardScaler(),
                          DecisionTreeClassifier(criterion='gini',max_depth=7,random_state=1))
pipe_tree.fit(X_train, y_train)
y_pred_tree= pipe_tree.predict(X_valid)
print('    Training accuracy of tree:', pipe_tree.score(X_train, y_train))
print('    Valid Accuracy of tree: %.8f' % pipe_tree.score(X_valid, y_valid))
print('    AUC value of tree : ',AUC_calculate(pipe_tree,valid,X_valid))
print('End tree model .')


# SGDC
print('Start sgdc model :')
from sklearn.linear_model import SGDClassifier 
pipe_sgdc = make_pipeline(StandardScaler(),SGDClassifier(loss='modified_huber',penalty = 'elasticnet'))# loss:hinge,log,modified_huber ; penalty: l1,l2,elasticnet
pipe_sgdc.fit(X_train, y_train)
y_pred_sgdc = pipe_sgdc.predict(X_valid)
print('    Training accuracy of sgdc :', pipe_sgdc.score(X_train, y_train))
print('    Valid Accuracy : %.8f' % pipe_sgdc.score(X_valid, y_valid))
print('    AUC value of sgdc : ',AUC_calculate(pipe_sgdc,valid,X_valid))
print('End sgdc model .')

## LGB
print('Start lgb model :')       
import lightgbm as lgb

lgb_ = lgb.LGBMClassifier(
                    learning_rate = 0.005,
                    boosting_type = 'gbdt',
                    objective = 'binary',
                    metric = 'logloss',
                    max_depth = 7,
                    sub_feature = 0.7,
                    num_leaves = 10,
                    colsample_bytree = 0.7,
                    min_data_in_leaf =10,
                    n_estimators = 500,
                    early_stop = 50,
                    verbose = -1,
                    feature_fraction= 0.7)

pipe_lgb = make_pipeline(StandardScaler(),lgb_)
                          
pipe_lgb.fit(X_train, y_train)
y_pred_lgb = pipe_lgb.predict(X_valid)
print('    Training accuracy of lgb: ', pipe_lgb.score(X_train, y_train))
print('    Valid Accuracy of lgb :  %.10f' % pipe_lgb.score(X_valid, y_valid))
print('    AUC value of lgb  : ',AUC_calculate(pipe_lgb,valid,X_valid))
print('End lgb model .')
