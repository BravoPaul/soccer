import pandas as pd




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
home_bet  
goals_bet  
footGoal_bet  
footGoal_home_half  
footGoal_away_half  
footGoal_home  
footGoal_away

'''




matches = pd.read_csv('../data/output/巴西保杯.csv')



# 主队让球分析
matches['expect_goal'] = matches['home_goal']+matches['home_bet']

matches_home_bet_win = matches[(matches['expect_goal'] - matches['away_goal'])>0]
matches_home_bet_lose = matches[(matches['expect_goal'] - matches['away_goal'])<0]
matches_home_bet_tie = matches[(matches['expect_goal'] - matches['away_goal'])==0]

# print(len(matches_home_bet_win))
# print(len(matches_home_bet_lose))
# print(len(matches_home_bet_tie))
# print(len(matches_home_bet_win)/(len(matches_home_bet_win)+len(matches_home_bet_lose)))


# 主场大球分析
matches['total_goals'] = matches['home_goal']+matches['away_goal']

matches_goal_bet_big = matches[matches['total_goals']>matches['goals_bet']]
matches_goal_bet_small = matches[matches['total_goals']<matches['goals_bet']]
matches_goal_bet_tie = matches[matches['total_goals']==matches['goals_bet']]
#
print(len(matches_goal_bet_big)) #40
print(len(matches_goal_bet_small)) #55
print(len(matches_goal_bet_tie)) #5
print(len(matches_goal_bet_small)/(len(matches_goal_bet_big)+len(matches_goal_bet_small))) #0.5789


def goals_bet_result(x,y):
    if x>y:
        return 1
    elif x==y:
        return 0
    else:
        return -1


matches['goals_bet_result'] = matches.apply(lambda x:goals_bet_result(x['total_goals'],x['goals_bet']),axis=1)
matches_goals_bet_g_hb = matches.groupby(['home_bet','goals_bet_result']).agg({"goals_bet_result": "count"}).rename(columns={'goals_bet_result': 'goals_bet_result_count'}).reset_index()

print(matches_goals_bet_g_hb)

matches_goals_bet_g_hb.to_csv('../data/output/anaylse_goals_巴西保杯.csv')
# matches_home_bet_win['']

# print(104/(95+104))


