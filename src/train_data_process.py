import pandas as pd
import os
import f_statistics as fs

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class TrainProcessor:
    # 传入一个赛季的数据
    def __init__(self, match_data):
        self.match_data = match_data

    def team_date_state(self,saiji):

        def columns_rename(x, rank, ad):
            if (x != 'home_team') & (x != 'date') & (x!='zk_flag'):
                return ad + '_' + str(rank) + '_' + x
            return x

        match_saiji = self.match_data[self.match_data['saiji']==saiji]
        index_o = match_saiji[match_saiji['zk_flag']==1][['home_team','date','away_team']]
        match_data_o = match_saiji[['home_team', 'date', 'home_goal', 'away_goal', 'home_yellow', 'away_yellow', 'home_red', 'away_red','footGoal_home', 'footGoal_away','score_last_season_home','score_new_season_home','score_last_season_away','score_new_season_away','zk_flag']]
        match_data_o_sorted = match_data_o.sort_values(by=['home_team','date'],ascending=True)
        start = 0
        match_max = match_data_o.groupby('home_team')['date'].count().max()
        feature_statisc_team = pd.DataFrame()
        feature_statisc_zk = pd.DataFrame()

        while True:
            if start==match_max-5:
                break
            data_pro = match_data_o_sorted.groupby(['home_team'],as_index=False).nth(list(range(start,start+5)))
            index_zk = match_data_o_sorted[['home_team','date','zk_flag']].groupby(['home_team'],as_index=False).nth(list(range(start+5,start+6)))
            index_team = match_data_o_sorted[['home_team','date']].groupby(['home_team'],as_index=False).nth(list(range(start+5,start+6)))

            feature_team = fs.f_team(data_pro)
            feature_team.columns = feature_team.columns.map(lambda x: columns_rename(x,5,'team'))
            feature_team.set_index('home_team',inplace=True)

            # 所有队伍坐镇主客场的信息
            feature_team_zk = fs.f_team(data_pro,0)
            feature_team_zk.columns = feature_team_zk.columns.map(lambda x: columns_rename(x, 5, 'zk'))

            # feature_team_rank = fs.f_rank_team(data_pro)
            # feature_team_rank.columns = feature_team_rank.columns.map(lambda x: columns_rename(x, 5, 'rank'))
            # feature_team_rank.set_index('home_team',inplace=True)

            f_temp = pd.merge(index, feature_team, how='inner', on=['home_team'])
            f_temp = pd.merge(f_temp, feature_team, how='inner', on=['home_team'])

            features = pd.merge(index,features,how='left',on=['home_team'])
            feature_statisc = pd.concat([feature_statisc,features],ignore_index=True)
            start = start+1


        feature_final = pd.merge(index_o,feature_statisc_team,how='inner',on=['home_team','date'])
        feature_statisc_team.columns = feature_statisc_team.columns.map(lambda x:columns_rename(x,'','a'))
        feature_statisc_team.rename(columns={'home_team': 'away_team'}, inplace=True)
        feature_final = pd.merge(feature_final,feature_statisc_team,how='inner',on=['away_team','date'])

        # feature_statisc_zk_home = feature_statisc_zk[feature_statisc_zk['zk_flag']==1]
        # del feature_statisc_zk_home['zk_flag']
        # feature_final = pd.merge(feature_final, feature_statisc_zk_home, how='inner', on=['home_team', 'date'])
        # feature_statisc_zk_away = feature_statisc_zk[feature_statisc_zk['zk_flag']==0]
        # del feature_statisc_zk_away['zk_flag']
        # feature_statisc_zk_away.columns = feature_statisc_zk_away.columns.map(lambda x: columns_rename(x, '', 'a'))
        # feature_statisc_zk_away.rename(columns={'home_team': 'away_team'}, inplace=True)
        # feature_final = pd.merge(feature_final, feature_statisc_zk_away, how='inner', on=['away_team', 'date'])
        feature_final['saiji'] = saiji
        feature_final.to_csv('../data/features/f_state/f_states_'+str(saiji)+'.csv')




    def train_data_final(self):
        origin_data = pd.read_csv('/Users/kunyue/project_personal/soccer/data/英超/soccer/game_end.csv')

        def label_go(row):
            if row['home_goal'] > row['away_goal']:
                return 0
            elif row['home_goal'] < row['away_goal']:
                return 2
            else:
                return 1


        fire_list = list(os.walk('../data/features/f_state'))
        dir = fire_list[0][0]
        files = fire_list[0][2]
        train_data = pd.DataFrame()
        for file in files:
            result = pd.read_csv(dir + '/' + file)
            train_data = pd.concat([train_data, result], sort=False, axis=0)
        origin_data_f = origin_data[['home_team','away_team','date','home_goal','away_goal']]
        train_data = pd.merge(train_data,origin_data_f,how='left',on=['home_team','away_team','date'])
        train_data['win_lose_label'] = train_data.apply(lambda row: label_go(row),axis=1)
        del train_data['home_goal']
        del train_data['away_goal']
        train_data.to_csv('../data/train_data/train_data.csv')

# 主客队主客场有问题



match_data = pd.read_csv('../data/import_p_data/yc_duplicate_data_with_rank.csv')
haha = TrainProcessor(match_data)
haha.team_date_state(18)
haha.team_date_state(17)
haha.team_date_state(16)
haha.team_date_state(15)
haha.team_date_state(14)
haha.train_data_final()