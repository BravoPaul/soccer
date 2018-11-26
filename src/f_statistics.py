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
        grp_home = data.groupby(['away_team'])
    home_attack_f = grp_home.agg({'home_goal':['mean','std','median','max'],'footGoal_home':['mean','std','median','max'],'home_yellow':['mean','max'],'home_red':'mean'})
    home_defend_f = grp_home.agg({'away_goal':['mean','std','median','max'],'footGoal_away':['mean','std','median','max'],'away_yellow':['mean','max'],'away_red':'mean'})
    home_attack_f.columns = ["_".join(x) for x in home_attack_f.columns.ravel()]
    home_defend_f.columns = ["_".join(x) for x in home_defend_f.columns.ravel()]
    home_attack_f = home_attack_f.reset_index()
    home_defend_f = home_defend_f.reset_index()
    home_f = pd.merge(home_attack_f, home_defend_f, how='inner', on=['home_team'])
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


def f_rank_away(data,index,rank=None):
    def columns_rename(x):
        if (x!='away_team') & (x!='date_index'):
           return 'away_'+str(rank)+'_'+x
        return x
    data_totals = data[['away_team','date','home_goal','away_goal','home_yellow','away_yellow','home_red','away_red','footGoal_home','footGoal_away','last_split_score_home','last_split_score_away']]
    temp_total = pd.merge(index,data_totals,how='left',on=['away_team'])
    temp_total = temp_total[temp_total['date_index']>temp_total['date']]
    if rank is not None:
        temp_total = temp_total.sort_values(by=['away_team','date'],ascending=False)
        temp_total = temp_total.groupby(['away_team'],as_index=False).nth(list(range(rank)))
    max_score = temp_total['last_split_score_away'].max()
    min_score = temp_total['last_split_score_away'].min()
    minus_score = max_score-min_score
    temp_total['net_score'] = minus_score - (temp_total['last_split_score_away'] - temp_total['last_split_score_home'])
    temp_total['net_goal'] = temp_total['away_goal'] - temp_total['home_goal']
    temp_total['net_foot_goal'] = temp_total['footGoal_away'] - temp_total['footGoal_home']
    temp_total['rank_goal'] = temp_total['net_score']*temp_total['net_goal']
    temp_total['rank_foot_goal'] = temp_total['net_score']*temp_total['net_foot_goal']
    temp_total['rank_goal_norm'] = temp_total['net_score']*(temp_total['net_goal'] - temp_total['net_goal'].min())/(temp_total['net_goal'].max()-temp_total['net_goal'].min())

    grp_home = temp_total.groupby(['away_team','date_index'])
    rank_f = grp_home.agg({'rank_goal':['mean','std','median','max'],'rank_foot_goal':['mean','std','median','max'],'rank_goal_norm':['mean','std','max']})
    rank_f.columns = ["_".join(x) for x in rank_f.columns.ravel()]
    rank_f.columns = rank_f.columns.map(lambda x:columns_rename(x))
    return rank_f.reset_index()


def label_total_goal(index,data):
    def label_go(x):
        if (x>=0) and (x<=1):
            return 0
        elif (x>=2) and (x<=3):
            return 1
        elif (x >= 4) and (x<=6):
            return 3
        else:
            return 4
    index = index.rename(columns={'date_index': 'date'})
    index_data = pd.merge(index,data,how='left',on=['home_team','away_team','date'])
    label_data = (index_data['home_goal']+index_data['away_goal']).map(label_go)
    return label_data


def label_win_lose(index,data):
    def label_go(row):
        if row['home_goal']>row['away_goal']:
            return 0
        elif row['home_goal']<row['away_goal']:
            return 1
        else:
            return 2
    index = index.rename(columns={'date_index': 'date'})
    index_data = pd.merge(index,data,how='left',on=['home_team','away_team','date'])
    index_data['label'] = index_data.apply(lambda x: label_go(x),axis = 1)
    return index_data['label']


def train_data_maker(file):
    data = pd.read_csv(file)
    minus_day = data['minus_day'].values[0]
    dure_day = data['dure'].values[0]
    data_o = data[data['zk_flag'] == 1]
    index_data_o = data_o[['home_team', 'away_team', 'date']]
    index_data = index_data_o.sort_values(by=['date'], ascending=False).head(20)
    date_min = index_data['date'].min()
    index_data_min = index_data_o[index_data_o['date'] == date_min]
    index = pd.concat([index_data, index_data_min], ignore_index=True).drop_duplicates()
    index.rename(columns={'date': 'date_index'}, inplace=True)

    feature_home_none = f_home(data,index,5)
    feature_home_3 = f_home(data,index,3).drop(['home_team','date_index'],axis=1)
    feature_home_zk = f_home_zk(data,index,3).drop(['home_team','date_index'],axis=1)
    feature_rank_home_3 = f_rank_home(data,index,3).drop(['home_team','date_index'],axis=1)

    feature_home = pd.concat([feature_home_none, feature_home_3,feature_home_zk,feature_rank_home_3],axis=1)

    feature_away_none = f_away(data,index,5)
    feature_away_3 = f_away(data,index,3).drop(['away_team','date_index'],axis=1)
    feature_away_zk = f_away_zk(data,index,3).drop(['away_team','date_index'],axis=1)
    feature_rank_away_3 = f_rank_away(data,index,3).drop(['away_team','date_index'],axis=1)
    feature_away = pd.concat([feature_away_none, feature_away_3,feature_away_zk,feature_rank_away_3],axis=1)

    train_data_temp = pd.merge(index, feature_home, on=['home_team', 'date_index'], how='left')
    train_data = pd.merge(train_data_temp, feature_away, on=['away_team', 'date_index'], how='left')

    l_total_goal = label_total_goal(index,data_o)
    l_win_lose = label_win_lose(index,data_o)

    train_data['total_goal_label'] = l_total_goal
    train_data['win_lose_label'] = l_win_lose

    return train_data



if __name__ == "__main__":
    fire_list = list(os.walk('/Users/kunyue/project_personal/soccer/data/split_data'))
    dir = fire_list[0][0]
    files = fire_list[0][2]
    train_data = pd.DataFrame()
    for file in files:
        result = train_data_maker(dir+'/'+file)
        train_data = pd.concat([train_data, result], sort=False, axis=0)
    train_data.to_csv('/Users/kunyue/project_personal/soccer/data/features/f_score/scores.csv')




