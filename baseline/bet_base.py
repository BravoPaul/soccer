import pandas as pd
import sqlite3
import numpy as np

pd.set_option('display.width', None)

with sqlite3.connect('/Users/kunyue/Documents/soccer_pro/data/database.sqlite') as engine:
    matches = pd.read_sql_query("SELECT * FROM Match "
                                "where league_id = 1729 "
                                "and season in"
                                "('2010/2011', '2011/2012', '2012/2013', '2013/2014', '2014/2015', '2015/2016')"
                                , engine)

    teams = pd.read_sql_query("SELECT * FROM Team", engine)[['team_api_id', 'team_long_name']]
    matches_small = matches[['date', 'home_team_api_id', 'away_team_api_id', 'B365H', 'B365D', 'B365A']]

f_matches = pd.merge(pd.merge(matches_small, teams, how='left', left_on='home_team_api_id', right_on='team_api_id'),
                     teams, how='left', left_on='away_team_api_id', right_on='team_api_id')

f_matches['date'] = f_matches['date'].map(lambda x: x.strip(" ")[0:10])


def bet_predict(home,tie,away):
    return np.argmax([away,tie,home]) - 1


f_matches['bet_pred'] = f_matches.apply(lambda x:bet_predict(x['B365H'],x['B365D'],x['B365A']),axis=1)
print(f_matches)