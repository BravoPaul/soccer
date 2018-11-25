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
比分特征主要从这么几个方面来刻画主客队的信息：

时间间隔

1，主队（客队）进攻能力
1.1，球队算数平均进攻能力（得分和角球平均值）
1.2，球队加权平均进攻能力（考虑到对面球队的实力）
1.3，球队目前最大的进攻能力（得分最大值）
1.4，球队加权平均最大的进攻能力（考虑到对面球队的实力）

2，主队（客队）防守能力
1.1，对方球队算数平均进攻能力（得分和角球平均值）
1.2，对方球队加权平均进攻能力（考虑到对面球队的实力）
1.3，对方球队目前最大的进攻能力（得分最大值）
1.4，对方球队加权平均最大的进攻能力（考虑到对面球队的实力）

3，主客队防守吃牌能力
4，主队（客队）爆冷概率程度
5，主队（客队）强弱对心理加成
6，主客场
7，球队伤残，稳定性（暂时不做）
8，求战欲望（这个不做）
'''



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

