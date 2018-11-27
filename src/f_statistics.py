import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os


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
4，主队（客队）爆冷概率程度  .........
5，主队（客队）强弱对心理加成
6，主客场
7，球队伤残，稳定性（暂时不做）
8，求战欲望（这个不做）
'''

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


# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)




# 主队球队平均总进球数，平均总失球数，总的最大进球数，总的最大失球数，方差，中位数，最小进球数,吃牌,造牌
def f_team(data,zk = 1):
    if zk==1:
        grp_home = data.groupby(['home_team'])
    else:
        grp_home = data.groupby(['home_team','zk_flag'])
    home_attack_f = grp_home.agg({'home_goal':['mean','std','max']})
    home_defend_f = grp_home.agg({'away_goal':['mean','std','max']})
    home_attack_f.columns = ["_".join(x) for x in home_attack_f.columns.ravel()]
    home_defend_f.columns = ["_".join(x) for x in home_defend_f.columns.ravel()]
    home_attack_f = home_attack_f.reset_index()
    home_defend_f = home_defend_f.reset_index()
    if zk==1:
        home_f = pd.merge(home_attack_f, home_defend_f, how='inner', on=['home_team'])
    else:
        home_f = pd.merge(home_attack_f, home_defend_f, how='inner', on=['home_team','zk_flag'])
    return home_f


# 考虑到上赛季的排名的特征
def f_rank_team(data):
    data['net_goal'] = (data['home_goal'] / (data['home_goal'] + data['away_goal']+0.000001)).fillna(0)
    data['net_score_old'] = (data['score_last_season_home'] / (data['score_last_season_home'] + data['score_last_season_away']+0.000001)).fillna(0)
    data['net_score_new'] = (data['score_new_season_home'] / (data['score_new_season_home'] + data['score_new_season_away']+0.000001)).fillna(0)
    data['net_score_new'] = data['net_score_new'].map(lambda x: 1 if x == 0 else x)
    data['net_score_old'] = data['net_score_old'].map(lambda x: 1 if x == 0 else x)
    data['old_rank'] = (data['net_goal']/(data['net_score_old'])).fillna(0)
    data['new_rank'] = (data['net_goal']/(data['net_score_new'])).fillna(0)
    grp_home = data.groupby(['home_team'])
    rank_f = grp_home.agg({'old_rank':['mean','std','median','max'],'new_rank':['mean','std','median','max',],'net_goal':['mean','std','max']})
    rank_f.columns = ["_".join(x) for x in rank_f.columns.ravel()]
    return rank_f.reset_index()


