import pandas as pd
import datetime

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Indexer:
    # 传入一个赛季的数据
    def __init__(self, match_data,rank_data):
        self.now_date = datetime.datetime.now()
        self.season_start_18 = datetime.datetime.strptime('18/08/12', '%y/%m/%d')
        self.match_data = match_data
        # self.match_data['date'] = self.match_data['date'].map(lambda x: datetime.datetime.strptime(x, '%y/%m/%d'))
        self.rank_data = rank_data

    # 把前minus day的排名做成上赛季的排名，93天的数据为一个数据块，过一个月然后再做成一个数据块
    def split_data(self, season,m_day,delta_day):
        data_o = self.match_data[self.match_data['saiji']==season]
        season_start = data_o['date'].min()
        season_end = data_o['date'].max()
        start = datetime.datetime.strptime(season_start, '%y/%m/%d')
        end = datetime.datetime.strptime(season_end, '%y/%m/%d')
        minus_day = m_day
        rank_data_o = self.rank_data[self.rank_data['saiji'] == season]
        while True:
            if end > start + datetime.timedelta(days=minus_day):
                start_index = start + datetime.timedelta(days=minus_day)
                data = data_o[(data_o['date'] >= start.strftime('%y/%m/%d')) & (data_o['date'] < start_index.strftime('%y/%m/%d'))]
                if start.strftime('%y/%m/%d')!=season_start:
                    rank_data = rank_data_o[rank_data_o['date'] >= start.strftime('%y/%m/%d')]
                    max_date = rank_data['date'].min()
                    rank_data = rank_data[rank_data['date'] == max_date]
                    rank_data = rank_data[['home_team', 'score_new_season']]
                    rank_data_home = rank_data.rename(columns={'score_new_season': 'last_split_score_home'})
                    data = pd.merge(data,rank_data_home,how='left',on=['home_team'])
                    rank_data_away = rank_data.rename(columns={'score_new_season': 'last_split_score_away','home_team':'away_team'})
                    data = pd.merge(data, rank_data_away, how='left', on=['away_team'])
                else:
                    data['last_split_score_home'] = data['score_last_season_home']
                    data['last_split_score_away'] = data['score_last_season_away']
                min_score = data['last_split_score_home'].min()
                data['dure'] = (start - datetime.datetime.strptime(season_start,'%y/%m/%d')).days
                data['minus_day'] = minus_day
                data['last_split_score_home'].fillna(min_score,inplace=True)
                data['last_split_score_away'].fillna(min_score,inplace=True)
                data.to_csv('../data/split_data/data_' + str(season) + '_' + start.strftime('%y/%m/%d').replace('/','-') + '_' + start_index.strftime('%y/%m/%d').replace('/','-') + '_data.csv')
                start = start + datetime.timedelta(days=delta_day)
            else:
                break



if __name__ == "__main__":
    match_data = pd.read_csv('../data/import_p_data/yc_duplicate_data_with_rank.csv')
    rank_data = pd.read_csv('../data/import_p_data/rank_day.csv')
    haha = Indexer(match_data,rank_data)
    # m_day = 20
    # delta_day = m_day/2
    # while(True):
    #     if m_day>90:
    #         break
    #     else:
    #         haha.split_data(18,m_day,delta_day)
    #         haha.split_data(17,m_day,delta_day)
    #         haha.split_data(16,m_day,delta_day)
    #         haha.split_data(15,m_day,delta_day)
    #         haha.split_data(14,m_day,delta_day)
    #     m_day = m_day+10
    #     delta_day = m_day/2

    haha.split_data(18,90,20)
    haha.split_data(18,90,30)
    haha.split_data(17,90,20)
    haha.split_data(17,90,30)
    haha.split_data(16,90,20)
    haha.split_data(16,90,30)
    haha.split_data(15,90,20)
    haha.split_data(15,90,30)
    haha.split_data(14,90,20)
    haha.split_data(14,90,30)






# 校验数据

#

# self.season_start_17 = datetime.datetime.strptime('17/08/13', '%y/%m/%d')
# self.season_start_16 = datetime.datetime.strptime('16/08/14', '%y/%m/%d')
# self.season_start_15 = datetime.datetime.strptime('15/08/09', '%y/%m/%d')
# self.season_start_14 = datetime.datetime.strptime('14/08/14', '%y/%m/%d')
