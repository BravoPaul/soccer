import pandas as pd
import datetime

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class TrainProcessor:
    # 传入一个赛季的数据
    def __init__(self, match_data,rank_data):
        self.now_date = datetime.datetime.now()
        self.season_start_18 = datetime.datetime.strptime('18/08/12', '%y/%m/%d')
        self.match_data = match_data
        # self.match_data['date'] = self.match_data['date'].map(lambda x: datetime.datetime.strptime(x, '%y/%m/%d'))
        self.rank_data = rank_data

    def columns_rename(x):
        if (x != 'home_team') & (x != 'date_index'):
            return 'home_' + str(rank) + '_' + x
        return x

    def team_date_state(self,saiji):
        index = self.match_data[self.match_data['saiji']==saiji][['home_team','date']].drop_duplicates()



        data_totals = data[['home_team', 'date', 'home_goal', 'away_goal', 'home_yellow', 'away_yellow', 'home_red', 'away_red','footGoal_home', 'footGoal_away']]
        temp_total = pd.merge(index, data_totals, how='left', on=['home_team'])
        temp_total = temp_total[temp_total['date_index'] > temp_total['date']]



match_data = pd.read_csv('../data/import_p_data/yc_duplicate_data_with_rank.csv')
rank_data = pd.read_csv('../data/import_p_data/rank_day.csv')
haha = TrainProcessor(match_data,rank_data)
haha.team_date_state(17)