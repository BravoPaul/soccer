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
home_rank
away_rank

'''

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.width', None)

yc_match_data = pd.read_csv("../data/英超/英超.csv")


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


# 比赛本身的信息，基础特征
def util_match_ha(pd_all, total_match, home_away):
    pd_goal = pd_all.sort_values(by=home_away + ['date'])
    grp = pd_goal.groupby(home_away)

    def get_last_f(x):
        result = []
        for item in range(total_match):
            temps = np.nan_to_num(x.apply(lambda i: i.shift(item + 1)).values)
            result.append(temps)
        return pd.Series(np.sum(result, axis=0) / total_match, index=pd_goal.index)

    list_f = [grp['home_goal'], grp['away_goal'],
              grp['home_yellow'], grp['home_red'],
              grp['away_yellow'], grp['away_red'],
              grp['home_goal_half'], grp['away_goal_half'],
              grp['footGoal_home_half'], grp['footGoal_away_half'],
              grp['footGoal_home'], grp['footGoal_away']
              ]

    result_f = list(map(get_last_f, list_f))
    pd_goal['f_home_goal'] = result_f[0]
    pd_goal['f_away_goal'] = result_f[1]
    pd_goal['f_home_yellow'] = result_f[2]
    pd_goal['f_home_red'] = result_f[3]
    pd_goal['f_away_yellow'] = result_f[4]
    pd_goal['f_away_red'] = result_f[5]
    pd_goal['f_home_goal_half'] = result_f[6]
    pd_goal['f_away_goal_half'] = result_f[7]
    pd_goal['f_footGoal_home_half'] = result_f[8]
    pd_goal['f_footGoal_away_half'] = result_f[9]
    pd_goal['f_footGoal_home'] = result_f[10]
    pd_goal['f_footGoal_away'] = result_f[11]
    pd_goal['f_net_goal'] = pd_goal['f_home_goal'] - pd_goal['f_away_goal']
    pd_goal['f_net_yellow'] = pd_goal['f_home_yellow'] - pd_goal['f_away_yellow']
    pd_goal['f_net_goal_half'] = pd_goal['f_home_goal_half'] - pd_goal['f_away_goal_half']
    pd_goal['f_net_footGoal_half'] = pd_goal['f_footGoal_home_half'] - pd_goal['f_footGoal_away_half']
    pd_goal['f_net_footGoal'] = pd_goal['f_footGoal_home'] - pd_goal['f_footGoal_away']

    result = []
    for item in range(total_match):
        pd_goal['f_date'] = grp['date'].apply(lambda i: i.shift(item + 1))
        temps = np.nan_to_num(((pd.to_datetime(pd_goal['date'], format='%y/%m/%d') - pd.to_datetime(pd_goal['f_date'],
                                                                                                    format='%y/%m/%d')).map(
            lambda x: x.days)).values)
        result.append(temps)
    pd_goal['f_diff_time'] = pd.Series(np.sum(result, axis=0) / total_match, index=pd_goal.index)
    del pd_goal['f_date']

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




def data_pre_trait(pd_origin_o, grp_list):
    pd_origin_o = pd_origin_o.sort_values(by=grp_list + ['date'], ascending=False)
    index_m = pd_origin_o[grp_list + ['date']].groupby(grp_list).head(1)

    def columns_name(x):
        if x.find('away') >= 0:
            return x.replace('away', 'home')
        if x.find('home') >= 0:
            return x.replace('home', 'away')
        return x

    index_m_reverse = index_m.copy()
    index_m_reverse.columns = index_m_reverse.columns.map(lambda x: columns_name(x))
    return index_m, index_m_reverse


def data_back_trait(pd_goal, ss):
    f_names = []

    def columns_traite(x):
        if x.find('f_') >= 0:
            x = x + '_' + ss
            f_names.append(x)
            return x
        else:
            return x

    pd_goal.columns = pd_goal.columns.map(columns_traite)
    return pd_goal[f_names]


def f_home_away(pd_real):
    index_m, index_m_reverse = data_pre_trait(pd_real, ['home_team', 'away_team'])

    # 主客队前一场
    f_a_h = util_match_ha(pd_real, 1, ['home_team', 'away_team']).fillna(0)
    houzhui = 'ha_1'
    f_1 = data_back_trait(f_a_h, houzhui)

    # 主客队前二场
    # f_a_h = util_match_ha(pd_real, 2, ['home_team', 'away_team']).fillna(0)
    # houzhui = 'ha_2'
    # f_2 = data_back_trait(f_a_h, houzhui)
    #
    # # 主客队前三场
    # f_a_h = util_match_ha(pd_real, 3, ['home_team', 'away_team']).fillna(0)
    # houzhui = 'ha_3'
    # f_3 = data_back_trait(f_a_h, houzhui)

    pd_feature_ha_r = pd.concat([index_m, f_1], axis=1).dropna()

    # 假数据制造
    # index_m_reverse_d = index_m_reverse.rename(columns={'date': 'date_index'})
    # pd_faux = pd.merge(index_m_reverse_d, pd_real, how='left', on=['home_team', 'away_team'])
    # pd_faux = pd_faux[pd_faux['date_index'] > pd_faux['date']]
    # pd_faux = pd.concat([pd_faux, index_m_reverse], ignore_index=True, sort=False).fillna(0)
    # del pd_faux['date_index']
    # pd_faux_index = pd_faux[['home_team', 'away_team', 'date']]
    #
    # # 主客队调换前一场
    # f_a_h = util_match_ha(pd_faux, 1, ['home_team', 'away_team']).fillna(0)
    # houzhui = 'j_ha_1'
    # f_j_1 = data_back_trait(f_a_h, houzhui)
    #
    # pd_feature_ha_f = pd.concat([pd_faux_index, f_j_1], axis=1).dropna()
    # pd_feature_ha_f['temp_team'] = pd_feature_ha_f['home_team']
    # pd_feature_ha_f['home_team'] = pd_feature_ha_f['away_team']
    # pd_feature_ha_f['away_team'] = pd_feature_ha_f['temp_team']
    # del pd_feature_ha_f['temp_team']
    #
    # pd_feature = pd.merge(pd_feature_ha_r, pd_feature_ha_f, how='left', on=['home_team', 'away_team', 'date'])

    return pd_feature_ha_r


def f_home(pd_real):
    index_m, index_m_reverse = data_pre_trait(pd_real, ['home_team', 'away_team'])
    # 主队前一场
    f_h = util_match_ha(pd_real, 1, ['home_team']).fillna(0)
    houzhui = 'h_1'
    f_r_1 = data_back_trait(f_h, houzhui)
    # 主队前三场
    f_h = util_match_ha(pd_real, 3, ['home_team']).fillna(0)
    houzhui = 'h_3'
    f_r_3 = data_back_trait(f_h, houzhui)
    # 主队前5场
    f_h = util_match_ha(pd_real, 5, ['home_team']).fillna(0)
    houzhui = 'h_5'
    f_r_5 = data_back_trait(f_h, houzhui)

    pd_feature_ha_r = pd.concat([index_m, f_r_1, f_r_3, f_r_5], axis=1).dropna()
    del pd_feature_ha_r['away_team']

    # 假数据制造
    # index_m_reverse_d = index_m_reverse.rename(columns={'date': 'date_index'})
    # pd_faux = pd.merge(index_m_reverse_d, pd_real, how='left', on=['away_team'])
    # pd_faux = pd_faux[pd_faux['date_index'] > pd_faux['date']]
    # pd_faux = pd.concat([pd_faux, index_m_reverse], ignore_index=True, sort=False).fillna(0)
    # del pd_faux['date_index']
    # pd_faux_index = pd_faux[['away_team', 'date']]
    #
    # # 主队作为客队前一场
    # f_a_h = util_match_ha(pd_faux, 1, ['away_team']).fillna(0)
    # houzhui = 'j_h_1'
    # f_j_1 = data_back_trait(f_a_h, houzhui)
    #
    # f_a_h = util_match_ha(pd_faux, 1, ['away_team']).fillna(0)
    # houzhui = 'j_h_2'
    # f_j_3 = data_back_trait(f_a_h, houzhui)
    #
    # f_a_h = util_match_ha(pd_faux, 1, ['away_team']).fillna(0)
    # houzhui = 'j_h_3'
    # f_j_5 = data_back_trait(f_a_h, houzhui)
    #
    # pd_feature_ha_f = pd.concat([pd_faux_index, f_j_1, f_j_3, f_j_5], axis=1).dropna()
    #
    # pd_feature_ha_f['home_team'] = pd_feature_ha_f['away_team']
    # del pd_feature_ha_f['away_team']
    # pd_feature = pd.merge(pd_feature_ha_r, pd_feature_ha_f, how='left', on=['home_team', 'date'])

    return pd_feature_ha_r


def f_away(pd_real):
    index_m, index_m_reverse = data_pre_trait(pd_real, ['home_team', 'away_team'])

    # 主队前一场
    f_h = util_match_ha(pd_real, 1, ['away_team']).fillna(0)
    houzhui = 'a_1'
    f_r_1 = data_back_trait(f_h, houzhui)

    # 主队前三场
    f_h = util_match_ha(pd_real, 3, ['away_team']).fillna(0)
    houzhui = 'a_3'
    f_r_3 = data_back_trait(f_h, houzhui)

    # 主队前5场
    f_h = util_match_ha(pd_real, 5, ['away_team']).fillna(0)
    houzhui = 'a_5'
    f_r_5 = data_back_trait(f_h, houzhui)

    pd_feature_ha_r = pd.concat([index_m, f_r_1, f_r_3, f_r_5], axis=1).dropna()
    del pd_feature_ha_r['home_team']

    # 假数据制造
    # index_m_reverse_d = index_m_reverse.rename(columns={'date': 'date_index'})
    # pd_faux = pd.merge(index_m_reverse_d, pd_real, how='left', on=['home_team'])
    # pd_faux = pd_faux[pd_faux['date_index'] > pd_faux['date']]
    # pd_faux = pd.concat([pd_faux, index_m_reverse], ignore_index=True, sort=False).fillna(0)
    # del pd_faux['date_index']
    # pd_faux_index = pd_faux[['home_team', 'date']]
    #
    # # 主队作为客队前一场
    # f_a_h = util_match_ha(pd_faux, 1, ['home_team']).fillna(0)
    # houzhui = 'j_a_1'
    # f_j_1 = data_back_trait(f_a_h, houzhui)
    #
    # f_a_h = util_match_ha(pd_faux, 1, ['home_team']).fillna(0)
    # houzhui = 'j_a_2'
    # f_j_3 = data_back_trait(f_a_h, houzhui)
    #
    # f_a_h = util_match_ha(pd_faux, 1, ['home_team']).fillna(0)
    # houzhui = 'j_a_3'
    # f_j_5 = data_back_trait(f_a_h, houzhui)
    #
    # pd_feature_ha_f = pd.concat([pd_faux_index, f_j_1, f_j_3, f_j_5], axis=1).dropna()
    #
    # pd_feature_ha_f['away_team'] = pd_feature_ha_f['home_team']
    # del pd_feature_ha_f['home_team']
    # pd_feature = pd.merge(pd_feature_ha_r, pd_feature_ha_f, how='left', on=['away_team', 'date'])

    return pd_feature_ha_r





def train():
    def main_bet_predictor(x):
        if x<0:
            return 1
        elif x>0:
            return -1
        else:
            return 0

    match_go = pd.read_csv('../data/temp/' + 'f_ha_h_a_goals_toal' + '.csv')
    match_last = pd.read_csv('../data/temp/' + 'f_ha_h_a_last_goals_toal' + '.csv')
    match = pd.concat([match_go,match_last],axis=0)
    label = match['label'].values

    # y_bet = match['main_bet'].map(main_bet_predictor).values

    del match['label']
    train_data = match.values

    # 多分类
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    for train_index, test_index in kf.split(train_data):
        xgboost_model = xgb.XGBClassifier().fit(train_data[train_index], label[train_index])
        # xgb.XGBRegressor
        # 预测结果
        pred = xgboost_model.predict(train_data[test_index])
        # 标准答案
        ground_truth = label[test_index]
    print("混淆矩阵:")
    print(confusion_matrix(ground_truth, pred))
    print("准确率：")
    print(metrics.accuracy_score(ground_truth, pred))

    print("--------------分割线---------------")

    # print("混淆矩阵:")
    # print(confusion_matrix(label, y_bet))
    # print("准确率：")
    # print(metrics.accuracy_score(label, y_bet))






def win_lose_label(x):
    if x['home_goal']>x['away_goal']:
        return 1
    elif x['home_goal']<x['away_goal']:
        return -1
    else:
        return 0


def big_small_label(x):
    if (x['home_goal']+x['away_goal'])-x['goals_bet']>0:
        return 1
    elif (x['home_goal']+x['away_goal'])-x['goals_bet']<0:
        return -1
    else:
        return 0

def goals_toal_label(x):
    goals = x['home_goal']+x['away_goal']
    if goals>=0 and goals<=1 :
        return 0
    elif goals>=2 and goals<=3:
        return 1
    else:
        return 2


#
# f_home_away_d = f_home_away(yc_match_data)
# f_home_d = f_home(yc_match_data)
# f_away_d = f_away(yc_match_data)
# f_temp = pd.merge(f_home_away_d, f_home_d, how='left', on=['home_team', 'date'])
# f_ha_h_a = pd.merge(f_temp, f_away_d, how='left', on=['away_team', 'date'])
#
# merge_data = pd.merge(f_ha_h_a, yc_match_data, how='left', on=['home_team', 'away_team', 'date'])
# merge_data['label'] = merge_data.apply(lambda row: goals_toal_label(row), axis=1)
# train_data = merge_data.drop(['home_team', 'away_team', 'date', 'id', 'match', 'home_goal', 'away_goal'], axis=1)
# train_data.to_csv('../data/temp/' + 'f_ha_h_a_goals_toal' + '.csv')
#
#
#
# # 上一个比赛季训练集制作
# pd_all = pd.read_csv("../data/英超/英超.csv")
# pd_goal = pd_all.sort_values(by=['home_team','away_team','date'],ascending=False)
# grp = pd_goal.groupby(['home_team','away_team']).nth(1).reset_index()
# last_matches = grp[['home_team','away_team','date']]
# last_matches.rename(columns={'date':'date_f'},inplace=True)
# yc_match_data_last = pd.merge(yc_match_data,last_matches,how='left',on=['home_team','away_team'])
# yc_match_data_last = yc_match_data_last[yc_match_data_last['date']<=yc_match_data_last['date_f']]
# del yc_match_data_last['date_f']
#
# f_home_away_d = f_home_away(yc_match_data_last)
# f_home_d = f_home(yc_match_data_last)
# f_away_d = f_away(yc_match_data_last)
# f_temp = pd.merge(f_home_away_d, f_home_d, how='left', on=['home_team', 'date'])
# f_ha_h_a = pd.merge(f_temp, f_away_d, how='left', on=['away_team', 'date'])
#
# merge_data = pd.merge(f_ha_h_a, yc_match_data_last, how='left', on=['home_team', 'away_team', 'date'])
# merge_data['label'] = merge_data.apply(lambda row: goals_toal_label(row), axis=1)
# train_data = merge_data.drop(['home_team', 'away_team', 'date', 'id', 'match', 'home_goal', 'away_goal'], axis=1)
# train_data.to_csv('../data/temp/' + 'f_ha_h_a_last_goals_toal' + '.csv')


train()




