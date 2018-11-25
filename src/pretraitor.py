import pandas as pd
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

yc_2013_score = [86, 84, 82, 79, 72, 69, 64, 56, 50, 49, 45, 42, 40, 38, 38, 37, 36, 33, 32, 30]
yc_2013_home_team = ['曼城', '利物浦','切尔西','阿森纳','埃弗顿','热刺','曼联','南安普敦','斯托克城','纽卡斯尔','水晶宫','斯旺西','西汉姆','桑德兰','阿斯顿维拉','赫尔','西布罗姆','诺维奇','富勒姆','卡迪夫']

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.width', None)

yc_match_data = pd.read_csv("../data/英超/英超.csv")


class Pretraitor():
    def __init__(self, df):

        df['saiji'] = df['date'].map(self.season_caculate)
        self.match_origin = df

        def home_away_change(x):
            if x.find('away') >= 0:
                return x.replace('away', 'home')
            if x.find('home') >= 0:
                return x.replace('home', 'away')
            return x

        match_temp = self.match_origin.copy()
        # match_temp.columns = sorted(match_temp.columns.map(lambda x: columns_name(x)))
        match_temp.columns = match_temp.columns.map(home_away_change)
        self.match_origin['zk_flag'] = 1
        match_temp['zk_flag'] = 0
        # self.match_origin.columns = sorted(self.match_origin.columns)
        self.match_duplicates = pd.concat([self.match_origin, match_temp], sort=True, ignore_index=True)

        self.match_duplicates['result_game'] = self.match_duplicates.apply(
            lambda x: self.result_game(x['home_goal'], x['away_goal']), axis=1)
        rank_team = self.match_duplicates.groupby(['home_team', 'saiji']).agg(
            {'result_game': 'mean'}).reset_index().rename(columns={'result_game': 'score_last_season'})
        rank_team['saiji'] = rank_team['saiji'] + 1
        score_2013 = pd.DataFrame({'home_team':yc_2013_home_team,'score_last_season':yc_2013_score,'saiji':14})
        score_2013['score_last_season'] = score_2013['score_last_season']/38
        rank_team = pd.concat([rank_team,score_2013],sort=True)
        self.team_rank = rank_team.drop_duplicates(subset=['home_team','saiji'])

    @classmethod
    def result_game(cls, team1_goal, team2_goal):
        result_defen = 1
        try:
            if team1_goal > team2_goal:
                result_defen = 3
            elif team1_goal < team2_goal:
                result_defen = 0
            return result_defen
        except TypeError:
            print(team1_goal)
            print(team2_goal)

    @classmethod
    def season_caculate(cls, x):
        if x > '14/08/00' and x < '15/06/00':
            season = 14
        elif x > '15/08/00' and x < '16/06/00':
            season = 15
        elif x > '16/08/00' and x < '17/06/00':
            season = 16
        elif x > '17/08/00' and x < '18/06/00':
            season = 17
        elif x > '18/08/00' and x < '19/06/00':
            season = 18
        else:
            raise ValueError
        return season

    def get_match_data_with_rank(self):
        rank_team_day = self.rank_change_day()
        rank_team_day = rank_team_day.rename(columns={'score_new_season':'score_new_season_home'})
        rank_team_season = self.team_rank.rename(columns={'score_last_season':'score_last_season_home'})
        match_temp = pd.merge(self.match_duplicates, rank_team_season, how='left',on=['home_team','saiji'])
        match_temp = pd.merge(match_temp,rank_team_day,how='left',on=['home_team','date'])
        rank_team_day = rank_team_day.rename(columns={'score_new_season_home':'score_new_season_away','home_team':'away_team'})
        rank_team_season = rank_team_season.rename(columns={'score_last_season_home':'score_last_season_away','home_team':'away_team'})
        match_temp = pd.merge(match_temp, rank_team_season, how='left', on=['away_team' ,'saiji'])
        match_done = pd.merge(match_temp, rank_team_day, how='left', on=['away_team','date'])
        match_done.sort_values(by=['date'],ascending=False).to_csv('../data/import_p_data/yc_duplicate_data_with_rank.csv')



    def rank_change_day(self):
        def one_season_jifen(match_saiji):
            biSaiRi = match_saiji['date'].values
            empty_df = None
            for value in biSaiRi:
                match_date = match_saiji[match_saiji['date'] < value].groupby(['home_team', 'saiji']).agg(
                    {'result_game': 'mean'}).reset_index()
                match_date['date'] = value
                if empty_df is None:
                    empty_df = match_date
                else:
                    empty_df = pd.concat([empty_df, match_date], ignore_index=True)
            empty_df.rename(columns={'result_game': 'score_new_season'}, inplace=True)
            return empty_df

        partial = self.match_duplicates[['home_team', 'result_game', 'date', 'saiji']]
        match_14 = partial[partial['saiji'] == 14]
        match_15 = partial[partial['saiji'] == 15]
        match_16 = partial[partial['saiji'] == 16]
        match_17 = partial[partial['saiji'] == 17]
        match_18 = partial[partial['saiji'] == 18]
        rank_team_go_14 = one_season_jifen(match_14)
        rank_team_go_15 = one_season_jifen(match_15)
        rank_team_go_16 = one_season_jifen(match_16)
        rank_team_go_17 = one_season_jifen(match_17)
        rank_team_go_18 = one_season_jifen(match_18)
        result = pd.concat([rank_team_go_14,rank_team_go_15,rank_team_go_16,rank_team_go_17,rank_team_go_18])
        result[['home_team', 'date','saiji' ,'score_new_season']].drop_duplicates(subset=['home_team', 'date']).to_csv('../data/import_p_data/rank_day.csv')
        return result[['home_team','date','score_new_season']].drop_duplicates(subset=['home_team','date'])



pre_obj = Pretraitor(yc_match_data)
pre_obj.rank_change_day()
# pre_obj.get_match_data_with_rank()
