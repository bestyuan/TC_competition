# coding=utf-8
# @author:bryan
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

ad_feature=pd.read_csv('./data/adFeature.csv')
if os.path.exists('./data/userFeature.csv'):
    user_feature=pd.read_csv('./data/userFeature.csv')
else:
    userFeature_data = []
    with open('./data/userFeature.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 100000 == 0:
                print(i)
        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv('./data/userFeature.csv', index=False)
train=pd.read_csv('./data/train.csv')
predict=pd.read_csv('./data/test.csv')
train.loc[train['label']==-1,'label']=0
predict['label']=-1
data=pd.concat([train,predict])
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')



# #年龄均值特征
grouped = data.groupby('aid')['age'].mean().reset_index()
grouped.columns = ['aid', 'aid_mean_age']
data = data.merge(grouped, how='left', on='aid')
grouped = data.groupby('adCategoryId')['age'].mean().reset_index()
grouped.columns = ['adCategoryId', 'adCategoryId_mean_age']
data = data.merge(grouped, how='left', on='adCategoryId')
add = pd.DataFrame(data.groupby('age')['adCategoryId'].nunique()).reset_index()
add.columns = ['age', 'age_active_adCategoryId']
data = data.merge(add, how='left', on=['age'])
grouped = data.groupby('LBS')['age'].mean().reset_index()
grouped.columns = ['LBS', 'position_mean_age']
data = data.merge(grouped, how='left', on='LBS')

grouped = data.groupby('uid')['creativeSize'].mean().reset_index()
grouped.columns = ['uid', 'uid_mean_creativeSize']
data = data.merge(grouped, how='left', on='uid')
grouped = data.groupby('uid')['adCategoryId'].mean().reset_index()
grouped.columns = ['uid', 'uid_mean_adCategoryId']
data = data.merge(grouped, how='left', on='uid')
#
#
#
# #广告的活跃特征
add = pd.DataFrame(data.groupby(["adCategoryId"]).aid.nunique()).reset_index()
add.columns = ["adCategoryId", "adCategoryId_active_app"]
data = data.merge(add, on=["adCategoryId"], how="left")
add = pd.DataFrame(data.groupby(["ct"]).aid.nunique()).reset_index()
add.columns = ["ct", "ct_active_app"]
data = data.merge(add, on=["ct"], how="left")
#
#
#
# #用户的活跃特征
add = pd.DataFrame(data.groupby(["aid"]).uid.nunique()).reset_index()
add.columns = ["aid", "aid_active_user"]
data = data.merge(add, on=["aid"], how="left")
add = pd.DataFrame(data.groupby(["adCategoryId"]).uid.nunique()).reset_index()
add.columns = ["adCategoryId", "adCategoryId_active_user"]
data = data.merge(add, on=["adCategoryId"], how="left")
add = pd.DataFrame(data.groupby(["uid"]).creativeId.nunique()).reset_index()
add.columns = ["uid", "user_active_creative"]
data = data.merge(add, on=["uid"], how="left")
add = pd.DataFrame(data.groupby(["LBS"]).uid.nunique()).reset_index()
add.columns = ["LBS", "LBS_active_uid"]
data = data.merge(add, on=["LBS"], how="left")
add = pd.DataFrame(data.groupby(["LBS"]).advertiserId.nunique()).reset_index()
add.columns = ["LBS", "LBS_active_advertiser"]
data = data.merge(add, on=["LBS"], how="left")
add = pd.DataFrame(data.groupby(["uid"]).creativeSize.nunique()).reset_index()
add.columns = ["uid", "user_active_creativeSize"]
data = data.merge(add, on=["uid"], how="left")
add = pd.DataFrame(data.groupby(["uid"]).aid.nunique()).reset_index()
add.columns = ["uid", "uid_active_aid"]
data = data.merge(add, on=["uid"], how="left")

add = pd.DataFrame(data.groupby(["interest1"]).creativeId.nunique()).reset_index()
add.columns = ["interest1", "interest1_active_creativeId"]
data = data.merge(add, on=["interest1"], how="left")
#
#
#
#
# #性别特征
add=pd.DataFrame(data.groupby(['gender']).adCategoryId.nunique()).reset_index()
add.columns=['gender','gender_active_adCategoryId']
data=data.merge(add,how='left',on=['gender'])
add=pd.DataFrame(data.groupby(['education']).adCategoryId.nunique()).reset_index()
add.columns=['education','education_active_adCategoryId']
data=data.merge(add,how='left',on=['education'])
# 活跃position数特征
add = pd.DataFrame(data.groupby(["aid"]).LBS.nunique()).reset_index()
add.columns = ["aid", "aid_active_LBS"]
data = data.merge(add, on=["aid"], how="left")
add = pd.DataFrame(data.groupby(["adCategoryId"]).LBS.nunique()).reset_index()
add.columns = ["adCategoryId", "adCategoryId_active_LBS"]
data = data.merge(add, on=["adCategoryId"], how="left")
#
# #交叉特征
data['age_adCategoryId']=data['age']*data['adCategoryId']
data['gender_creativeId']=data['gender']*data['creativeId']
data['age_creativeId']=data['age']*data['creativeId']
data['age_creativeSize']=data['age']*data['creativeSize']
data['gender_creativeSize']=data['gender']*data['creativeSize']
data['education_creativeSize']=data['education']*data['creativeSize']
data['gender_education_adCategoryId']=data['gender']*data['education']*data['adCategoryId']
grouped = data.groupby('productType')['age'].mean().reset_index()
grouped.columns = ['productType', 'productType_mean_age']
data = data.merge(grouped, how='left', on='productType')
grouped = data.groupby('creativeSize')['age'].mean().reset_index()
grouped.columns = ['creativeSize', 'creativeSize_mean_age']
data = data.merge(grouped, how='left', on='creativeSize')

data=data.fillna('-1')
one_hot_feature=['LBS','age','carrier','consumptionAbility','education',
                 'gender','house','ct','marriageStatus',
                 'advertiserId','campaignId', 'creativeId',
                'os','age_adCategoryId','gender_creativeId',
                 'age_creativeId','age_creativeSize','gender_creativeSize','education_creativeSize',
                'productType_mean_age', 'creativeSize_mean_age',
                'uid_mean_creativeSize',"user_active_creativeSize","uid_active_aid",'uid_mean_adCategoryId',
                ]

# one_hot_feature=['LBS','age','carrier','consumptionAbility','education',
#                  'gender','house','ct','marriageStatus',
#                  'advertiserId','campaignId', 'creativeId',
#                 'os','age_adCategoryId','age_creativeId','age_creativeSize',
#                 ]

vector_feature=['interest1','interest2','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
new_feature=[ 'aid_mean_age','adCategoryId_mean_age',"adCategoryId_active_app", "ct_active_app", "aid_active_user",
                 "adCategoryId_active_user","user_active_creative",'gender_active_adCategoryId',
                 'education_active_adCategoryId','age_active_adCategoryId',"aid_active_LBS","adCategoryId_active_LBS",
                 "LBS_active_advertiser","LBS_active_uid",'position_mean_age'
                ]


for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

for feature in new_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])



train=data[data.label!=-1]
train_y=train.pop('label')
# train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
test=data[data.label==-1]
res=test[['aid','uid']]
test=test.drop('label',axis=1)
enc = OneHotEncoder()
train_x=train[['creativeSize','aid_mean_age','adCategoryId_mean_age',"adCategoryId_active_app", "ct_active_app", "aid_active_user",
                 "adCategoryId_active_user","user_active_creative",'gender_active_adCategoryId',
                 'education_active_adCategoryId','age_active_adCategoryId',"aid_active_LBS","adCategoryId_active_LBS",
                 "LBS_active_advertiser","LBS_active_uid",'position_mean_age']]
# train_x=train[['creativeSize']]
test_x=test[['creativeSize','aid_mean_age','adCategoryId_mean_age',"adCategoryId_active_app", "ct_active_app", "aid_active_user",
                 "adCategoryId_active_user","user_active_creative",'gender_active_adCategoryId',
                 'education_active_adCategoryId','age_active_adCategoryId',"aid_active_LBS","adCategoryId_active_LBS",
                 "LBS_active_advertiser","LBS_active_uid",'position_mean_age'
             ]]
# test_x=test[['creativeSize']]
for feature in one_hot_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a=enc.transform(train[feature].values.reshape(-1, 1))
    test_a = enc.transform(test[feature].values.reshape(-1, 1))
    train_x= sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')


cv=CountVectorizer()
for feature in vector_feature:
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')





def LGB_test(train_x,train_y,test_x,test_y):
    from multiprocessing import cpu_count
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50,random_state=2018,n_jobs=cpu_count()-1
    )
    clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],eval_metric='auc',early_stopping_rounds=100)
    # print(clf.feature_importances_)
    return clf,clf.best_score_[ 'valid_1']['auc']

def LGB_predict(train_x,train_y,test_x,res):
    print("LGB test")
    # clf = lgb.LGBMClassifier(
    #     boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    #     max_depth=-1, n_estimators=8500, objective='binary',
    #     subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    #     learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=100
    # )
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=100, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=100
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
    res['score'] = clf.predict_proba(test_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('./data/submission.csv', index=False)
    os.system('./data/submission.csv')
    return clf

model=LGB_predict(train_x,train_y,test_x,res)