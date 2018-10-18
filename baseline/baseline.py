import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn_pandas import DataFrameMapper
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.width', None)

from subprocess import check_output
#
# with sqlite3.connect('/Users/kunyue/Documents/soccer_pro/data/database.sqlite') as engine:
#     matches = pd.read_sql_query("SELECT * FROM Match where league_id = 1729;", engine)

yc_match_data = pd.read_csv("/Users/kunyue/Documents/soccer_pro/data/yc_match_data.csv")

m_2013 = yc_match_data[yc_match_data['Date']>'2010-01-01']


#队伍特征
def f_onehot(pd_origin):
    return pd.get_dummies(pd_origin)


# 比分特征
def f_goal(pd_origin):
    pd_goal = pd_origin.sort_values(by=['HomeTeam', 'AwayTeam', 'Date'])
    pd_goal['l_home_goal'] = pd_goal.groupby(['HomeTeam', 'AwayTeam'])['FTHG'].apply(lambda i: i.shift(1))
    pd_goal['l_away_goal'] = pd_goal.groupby(['HomeTeam', 'AwayTeam'])['FTAG'].apply(lambda i: i.shift(1))
    pd_goal['l_Date'] = pd_goal.groupby(['HomeTeam', 'AwayTeam'])['Date'].apply(lambda i: i.shift(1))
    pd_goal['diff_time'] = (pd.to_datetime(pd_goal['Date']) - pd.to_datetime(pd_goal['l_Date'])).map(lambda x: x.days)
    return pd_goal[['Date', 'HomeTeam', 'AwayTeam', 'l_home_goal', 'l_away_goal', 'diff_time']]


def train_maker():
    pass


def label_maker(pd_origin):
    def label_go(a,b):
        if a-b > 0:
            return 1
        elif a-b == 0:
            return 0
        else:
            return -1
    pd_origin['label'] = pd_origin.apply(lambda x: label_go(x['FTHG'],x['FTAG']),axis=1)
    return pd_origin[['Date', 'HomeTeam', 'AwayTeam', 'label']]


# 时间特征
class DateEncoder(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dt = pd.DatetimeIndex(X.flatten())
        date_data = pd.DataFrame(np.transpose([dt.year, dt.month, dt.day]))
        return date_data


goal_fea = f_goal(m_2013)
label_data = label_maker(m_2013)
f_data = pd.merge(goal_fea, label_data)
td_onehot = f_onehot(f_data[['HomeTeam', 'AwayTeam']])
train_data = pd.concat([td_onehot, f_data], axis=1).fillna(0)

del train_data['AwayTeam']
del train_data['HomeTeam']

mapper = DataFrameMapper([
    (['Date'], DateEncoder()),

], default=None, df_out=True)

train_data_final = mapper.fit_transform(train_data)
print(train_data_final['label'].unique())

train_label_final = train_data_final['label'].values
del train_data_final['label']
train_pure_final = train_data_final.values

print("start training")
print(train_label_final)
print(train_pure_final)


# 多分类
y_iris = train_label_final
X_iris = train_pure_final
kf = KFold(n_splits=5, shuffle=True, random_state=1234)
for train_index, test_index in kf.split(X_iris):
    xgboost_model = xgb.XGBClassifier().fit(X_iris[train_index], y_iris[train_index])
    #预测结果
    pred = xgboost_model.predict(X_iris[test_index])
    #标准答案
    ground_truth = y_iris[test_index]
    print("混淆矩阵:", confusion_matrix(ground_truth, pred))
    print("准确率：", metrics.accuracy_score(ground_truth, pred))
