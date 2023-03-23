import pandas as pd
import numpy as np
from pyparsing import alphas
import seaborn as sns
from pylab import mpl
import lightgbm as lgb
import xgboost as xgb
from sklearn import preprocessing
from sklearn import svm 
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from bayes_opt import BayesianOptimization
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
data_path = './'
df_demand_train = pd.read_csv(data_path+'demand_train.csv',parse_dates = ['date'])
df_order_train  = pd.read_csv(data_path+'order_train.csv')
df_demand_test = pd.read_csv(data_path+'demand_test.csv',parse_dates=['date'])
df_order_test  = pd.read_csv(data_path+'order_test.csv')
df_order_train['type'] = df_order_train['type'].map({'A1':1,'A2':2,'A3':3})
df_order_test['type'] = df_order_test['type'].map({'A1':1,'A2':2,'A3':3})

# transformd datatype and other things
df_demand_train['year'] = df_demand_train['date'].apply(lambda x:x.year)
df_demand_train['month'] = df_demand_train['date'].apply(lambda x:x.month)
df_demand_train['day'] = df_demand_train['date'].apply(lambda x:x.day)

df_demand_test['year'] = df_demand_test['date'].apply(lambda x:x.year)
df_demand_test['month'] = df_demand_test['date'].apply(lambda x:x.month)
df_demand_test['day'] = df_demand_test['date'].apply(lambda x:x.day)

promt_month=df_demand_train.groupby(['product_id','year','month'])['is_sale_day'].sum()
df_order_train['promts'] = promt_month.values
promt_month=df_demand_test.groupby(['product_id','year','month'])['is_sale_day'].sum()
df_order_test['promts'] = promt_month.values


sale_month=df_demand_train.groupby(['product_id','year','month'])['label'].sum()
df_order_train['label'] = sale_month.values
df_order_test['label'] = 0

df_order_train_copy = df_order_train.copy()
df_order_test_copy = df_order_test.copy()

df_order_train.start_stock.fillna(0,inplace=True)
df_order_test.start_stock.fillna(0,inplace=True)
df_order_train.end_stock.fillna(0,inplace=True)
df_order_test.end_stock.fillna(0,inplace=True)

# make features

df_order_train['product_id']=df_order_train['product_id'].apply(lambda x:int(str(x)[1:]))
df_order_train['year']=df_order_train['year'].apply(lambda x:x-2018)
df_order_train['month_from_begin']=(df_order_train['year'].values)*12 + df_order_train['month']
df_order_train['is_month2']=(df_order_train['month']==2).apply(lambda x:(-1 if x else 1))   
df_order_train['masked_order']=(df_order_train['order'].values * ((df_order_train['label'] != 0) | df_order_train['month'].isin([10,11,12,1,2,3])).apply(lambda x:(1.0 if x else 0.2)))
df_order_train['masked_order'] = df_order_train['masked_order'].astype(np.float64)
df_order_train['masked_promts']=(df_order_train['promts'].values * ((df_order_train['label'] != 0) | df_order_train['month'].isin([10,11,12,1,2,3])).apply(lambda x:(1.0 if x else 0.2)))
df_order_train['masked_promts'] = df_order_train['masked_promts'].astype(np.float64)
df_order_train['cumsum_sales'] = np.log(df_order_train.groupby('product_id')['label'].cumsum()+1)
df_order_train['stock_dif'] = df_order_train.end_stock.values - df_order_train.start_stock.values

df_order_test['product_id']=df_order_test['product_id'].apply(lambda x:int(str(x)[1:]))
df_order_test['year']=df_order_test['year'].apply(lambda x:x-2018)
df_order_test['month_from_begin']=(df_order_test['year'].values)*12 + df_order_test['month']
df_order_test['is_month2']=(df_order_test['month']==2).apply(lambda x:(-1 if x else 1))   
df_order_test['masked_order']=df_order_test['order'].values
df_order_test['masked_promts']=df_order_test['promts'].values
df_order_test['cumsum_sales'] = df_order_train[(df_order_train['year']==3) & (df_order_train.month.isin([10,11,12]))]['cumsum_sales']
df_order_test['stock_dif'] = df_order_test.end_stock.values - df_order_test.start_stock.values

df_order_train['date'] = pd.to_datetime([
    '{}-{:02d}-01'.format(
    2018+df_order_train.year[i],df_order_train.month[i]
    ) for i in range(len(df_order_train))
])
df_order_test['date'] = pd.to_datetime([
    '{}-{:02d}-01'.format(
    2018+df_order_test.year[i],df_order_test.month[i]
    ) for i in range(len(df_order_test))
])


df_data = pd.concat((df_order_train,df_order_test),axis=0)
for i in ['3','6','9','12']:
    data_before = df_data.groupby('product_id')[['label','promts','start_stock']].shift(int(i)).fillna(0.)
    df_data[['before{}_label'.format(i),'before{}_promts'.format(i),'before{}_stock'.format(i)]] = data_before
# 统�?��?
df_data['before_mean_label'] = (df_data.before3_label.values + df_data.before6_label.values + df_data.before9_label.values)/3
df_data['before_mean_promts'] = (df_data.before3_promts.values + df_data.before6_promts.values + df_data.before9_promts.values)/3
df_data['before_dif1_label'] = (df_data.before3_label.values - df_data.before6_label.values)
df_data['before_dif2_label'] = (df_data.before6_label.values - df_data.before9_label.values)
df_data['before_dif3_label'] = (df_data.before3_label.values - df_data.before9_label.values)

added_feat = [
    'before3_label', 'before3_promts', 'before3_stock', 'before6_label', 'before6_promts','before6_stock',
    'before9_label', 'before9_promts', 'before9_stock', 'before12_label', 'before12_promts','before_mean_label',
    'before_mean_promts', 'before_dif1_label', 'before_dif2_label','before_dif3_label',]
df_order_train[added_feat] = df_data[added_feat].iloc[:len(df_order_train),:]
df_order_test[added_feat] = df_data[added_feat].iloc[len(df_order_test):,:]


feat_list = [
    'product_id', 
    'type', 
    'year', 
    'month', 
    'month_from_begin',
    # 'dayofmonth',
    'masked_order',
    'masked_promts', 
    # 'cumsum_sales',
    'start_stock',
    'end_stock',  
    'is_month2',
    'before3_label', 
    'before3_promts', 
    'before3_stock',
    'before6_label', 
    # 'before6_promts',
    # 'before6_stock',
    'before9_label', 
    # 'before9_promts',
    # 'before9_stock', 
    # 'before12_label', 
    # 'before12_promts',
    'before_mean_label',
    'before_mean_promts', 
    'before_dif1_label', 
    'before_dif2_label',
    # 'before_dif3_label'
    # 'label', 
]


df_train_X = df_order_train[(df_order_train.year.isin([1,2]))][feat_list]
df_train_y = df_order_train[(df_order_train.year.isin([1,2]))]['label']
df_test_X  = df_order_test[feat_list]

K = 9
kf = KFold(K,shuffle=True,random_state=520,)

#poisson regression

lgb_params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'poisson',
    'bagging_fraction': 0.8,
    'bagging_freq':1,
    'num_leaves': 80,
    'colsample_bytree': 0.45,
    'min_data': 200,
    'min_hessian': 1,   
    'verbose': -1
}

best_params = lgb_params.copy()

def evalerror(pred,df):
    label = df.get_label().copy()
    score = mean_squared_error(label,pred)*0.5
    return ('0.5 mse',np.sqrt(score),False)


def meanAcc(df_train:pd.DataFrame,preds:np.ndarray):
    df_train['preds'] = preds
    df_train.drop(df_train[df_train['product_id']==145].index,inplace=True)
    df_Acc = df_train[['year','month','product_id','label','preds']]
    
    gt = df_train['label'].values
    preds = df_Acc['preds'].values
    acc = (1-np.abs(gt-preds)/(gt+1))
    df_train['acc'] = acc
    df_train.head()
    
    gt = df_train.groupby(['year','month','product_id'])[['label']].mean().values
    acc = df_train.groupby(['year','month','product_id'])[['acc']].mean().values

    gt.shape,acc.shape
    NUM_CLASS = 208
    gt = gt.reshape((-1,NUM_CLASS))
    acc = acc.reshape((-1,NUM_CLASS))
    gt.shape,acc.shape
    gt = gt / (gt.sum(axis=1,keepdims=True))
    acc = (acc * gt).sum(axis=1)   

    # caculate the mean Acc
    year,month = 2020, 1
    mean_acc,weight = [],[]
    for i in range(acc.shape[0]):
        if month % 12 == 0:
            month = 0
            year += 1
        month +=1
        mean_acc.append(100*acc[i])
        weight.append((1.2 if month in [1,2,3] else 1.))

    mean_Acc = np.multiply(mean_acc,weight).sum() / sum(weight)

    return mean_Acc

best_mean_acc = 0.7

def bayes_function(**params):
    global best_mean_acc
    global best_params
    _params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'poisson',
        'bagging_fraction': 0.8,
        'bagging_freq':1,
        'num_leaves': 80,
        'colsample_bytree': 0.45,
        'min_data': 200,
        'min_hessian': 1,   
        'verbose': -1
    }

    # cast some value to int type
    for _k,_v in params.items():
        _params[_k] = int(_v) if _k in ['num_leaves','max_depth','min_data','bagging_freq'] else _v

    train_preds_lgb = np.zeros(df_train_X.shape[0])
    test_preds_lgb = np.zeros((df_test_X.shape[0],K))

    for i,(train_idx,val_idx) in enumerate(kf.split(df_train_X)):
        train_feat1 = df_train_X.iloc[train_idx]
        train_feat2 = df_train_X.iloc[val_idx]
        train_target1 = df_train_y.iloc[train_idx]
        train_target2 = df_train_y.iloc[val_idx]

        lgb_train1 = lgb.Dataset(train_feat1.values,train_target1.values)
        lgb_train2 = lgb.Dataset(train_feat2.values,train_target2.values)

        gbm = lgb.train(
            _params,
            lgb_train1,
            num_boost_round=20000,
            valid_sets=lgb_train2,
            verbose_eval=500,
            feval=evalerror,
            early_stopping_rounds=200,
        )
        train_preds_lgb[val_idx] += gbm.predict(train_feat2)
        test_preds_lgb[:,i] = gbm.predict(df_test_X)
        print('\n')
    
    mean_acc = meanAcc(df_order_train[df_order_train.year.isin([1,2])].copy(),train_preds_lgb)

    if best_mean_acc < mean_acc:
        best_mean_acc = mean_acc
        best_params = _params
        print('checkpoint ===>\n with best mean_acc:{:.4f}%'.format(mean_acc))

    return mean_acc

bounds_LGB = {
    'bagging_fraction': (0.5,1),
    'bagging_freq':(1,4),
    'num_leaves': (60,100),
    'colsample_bytree': (0.3,0.6),
    'min_data': (150,250),
    'min_hessian': (0.01,1),   
}

LGB_BO = BayesianOptimization(bayes_function,bounds_LGB,random_state=13)

print(LGB_BO.space.keys)

print('-'*100)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    best_mean_acc = 0.7
    LGB_BO.maximize(init_points=5,n_iter=40,acq='ucb',xi=0.,alpha=1e-6)

print(LGB_BO.max['params'])

with open('parmas.txt','w') as f:
    for _k,_v in best_params.items():
        f.write('{}:{}\n'.format(_k,_v))
