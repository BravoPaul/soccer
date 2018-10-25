import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn_pandas import DataFrameMapper
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import warnings

'''
date  
home_red  
home_yellow  
home_team  
home_goal  
away_goal  
away_team  
away_yellow  
away_red  
home_goal_half  
away_goal_half  
main_bet  
goals_bet  
footGoal_bet  
footGoal_home_half  
footGoal_away_half  
footGoal_home  
footGoal_away

'''

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.width', None)

yc_match_data = pd.read_csv("../data/英超.csv")


# 队伍特征
def f_onehot(pd_origin):
    return pd.get_dummies(pd_origin)


# 比分特征
def f_home_away(pd_origin):
    pd_goal = pd_origin.sort_values(by=['home_team', 'away_team', 'date'])
    pd_goal['l_home_goal'] = pd_goal.groupby(['home_team', 'away_team'])['home_goal'].apply(lambda i: i.shift(1))
    pd_goal['l_away_goal'] = pd_goal.groupby(['home_team', 'away_team'])['away_goal'].apply(lambda i: i.shift(1))
    pd_goal['l_net_goal'] = (pd_goal['l_home_goal'] - pd_goal['l_away_goal']) / (
            pd_goal['l_home_goal'] + pd_goal['l_away_goal'] + 0.0001)
    pd_goal['l_home_yellow'] = pd_goal.groupby(['home_team', 'away_team'])['home_yellow'].apply(lambda i: i.shift(1))
    pd_goal['l_away_yellow'] = pd_goal.groupby(['home_team', 'away_team'])['away_yellow'].apply(lambda i: i.shift(1))
    pd_goal['l_net_yellow'] = (pd_goal['l_home_yellow'] - pd_goal['l_away_yellow']) / (
            pd_goal['l_home_yellow'] + pd_goal['l_away_yellow'] + 0.0001)
    pd_goal['l_home_red'] = pd_goal.groupby(['home_team', 'away_team'])['home_red'].apply(lambda i: i.shift(1))
    pd_goal['l_away_red'] = pd_goal.groupby(['home_team', 'away_team'])['home_red'].apply(lambda i: i.shift(1))
    pd_goal['l_date'] = pd_goal.groupby(['home_team', 'away_team'])['date'].apply(lambda i: i.shift(1))
    pd_goal['diff_time'] = (pd.to_datetime(pd_goal['date']) - pd.to_datetime(pd_goal['l_date'])).map(lambda x: x.days)
    return pd_goal[['date', 'home_team', 'away_team', 'l_home_goal', 'l_away_goal', 'diff_time']]







def f_match_ha(pd_all,total_match,home_away):

    pd_goal = pd_all.sort_values(by=home_away+['date'])
    grp = pd_goal.groupby(home_away)

    def get_last_f(x):
        result = []
        for item in range(total_match):
            temps = np.nan_to_num(x.apply(lambda i: i.shift(item+1)).values)
            result.append(temps)
        return pd.Series(np.sum(result, axis=0) / total_match, index=pd_goal.index)

    list_f = [grp['home_goal'], grp['away_goal'],
              grp['home_yellow'], grp['home_red'],
              grp['away_yellow'], grp['away_red'],
              grp['flag']
              ]

    result_f = list(map(get_last_f, list_f))
    pd_goal['f_home_goal'] = result_f[0]
    pd_goal['f_away_goal'] = result_f[1]
    pd_goal['f_home_yellow'] = result_f[2]
    pd_goal['f_home_red'] = result_f[3]
    pd_goal['f_away_yellow'] = result_f[4]
    pd_goal['f_away_red'] = result_f[5]
    pd_goal['f_flag'] = result_f[6]
    pd_goal['f_net_goal'] = (pd_goal['f_home_goal'] - pd_goal['f_away_goal']) / (
            pd_goal['f_home_goal'] + pd_goal['f_away_goal'] + 0.0001)
    pd_goal['f_net_yellow'] = (pd_goal['f_home_yellow'] - pd_goal['f_away_yellow']) / (
            pd_goal['f_home_yellow'] + pd_goal['f_away_yellow'] + 0.0001)
    result = []
    for item in range(total_match):
        pd_goal['f_date'] = grp['date'].apply(lambda i: i.shift(item+1))
        temps = np.nan_to_num(((pd.to_datetime(pd_goal['date'], format='%y/%m/%d') - pd.to_datetime(pd_goal['f_date'],
                                                                                                    format='%y/%m/%d')).map(
            lambda x: x.days)).values)
        result.append(temps)
    pd_goal['f_diff_time'] = pd.Series(np.sum(result, axis=0) / total_match, index=pd_goal.index)

    return pd_goal




def train_maker():
    pass


def label_maker(pd_origin):
    def label_go(a, b):
        if a - b > 0:
            return 1
        elif a - b == 0:
            return 0
        else:
            return -1

    pd_origin['label'] = pd_origin.apply(lambda x: label_go(x['home_goal'], x['away_goal']), axis=1)
    return pd_origin[['date', 'home_team', 'away_team', 'label']]


# 时间特征
class dateEncoder(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dt = pd.datetimeIndex(X.flatten())
        date_data = pd.DataFrame(np.transpose([dt.year, dt.month, dt.day]))
        return date_data


def train():
    goal_fea = f_home_away(m_2013)
    label_data = label_maker(m_2013)
    f_data = pd.merge(goal_fea, label_data)
    td_onehot = f_onehot(f_data[['home_team', 'away_team']])
    train_data = pd.concat([td_onehot, f_data], axis=1).fillna(0)

    del train_data['away_team']
    del train_data['home_team']

    mapper = DataFrameMapper([
        (['date'], dateEncoder()),

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
        # 预测结果
        pred = xgboost_model.predict(X_iris[test_index])
        # 标准答案
        ground_truth = y_iris[test_index]
    print("混淆矩阵:", confusion_matrix(ground_truth, pred))
    print("准确率：", metrics.accuracy_score(ground_truth, pred))


def data_pre_trait(pd_origin_o,is_all=False):
    if not is_all:
        pd_origin_o['flag'] = 1
        return pd_origin_o
    def columns_name(x):
        if x.find('away') >= 0:
            return x.replace('away', 'home')
        if x.find('home') >= 0:
            return x.replace('home', 'away')
        return x
    pd_copy = pd_origin_o.copy()
    pd_copy.columns = pd_copy.columns.map(lambda x: columns_name(x))
    pd_origin_o['flag'] = 1
    pd_copy['flag'] = 0
    pd_all = pd.concat([pd_origin_o, pd_copy])
    return pd_all


def data_back_trait(pd_goal,ss):
    def columns_traite(x):
        if x.find('f_')>=0:
            return x+'_'+ ss
        else:
            return x
    pd_goal.columns = pd_goal.columns.map(columns_traite)
    return pd_goal


pd_all = data_pre_trait(yc_match_data)


f_a_h = f_match_ha(pd_all,3,['home_team']).fillna(0)

houzhui = 'all_h_3'

pd_feature = data_back_trait(f_a_h,houzhui)


pd_feature.to_csv('../data/temp/'+houzhui+'.csv')

# print(f_a_h)
