import datetime
import pandas as pd
import os


# c = pd.DataFrame()


print(list(os.walk('/Users/kunyue/project_personal/soccer/data/split_data')))

#
#
# p1 = {'home_team':['埃弗顿','埃弗顿','埃弗顿'],'away_team':['热刺','曼联','沃特福德'],'date':['17/10/14','17/11/01','17/11/06'],'home_goal':[2,3,1]}
# p2 = {'home_team':['埃弗顿','埃弗顿'],'away_team':['曼联','沃特福德'],'date_index':['17/11/01','17/11/06']}
#
#
# index = pd.DataFrame(p2)
# match = pd.DataFrame(p1)
#
# print(index)
#
# haha = pd.concat([c,index])
#
# print(haha)

#
# match = match[['home_team','date','home_goal']]
#
# haha = pd.merge(match,index,on=['home_team'],how='left')
#
# haha = haha[haha['date_index']>haha['date']]
#
# print(haha)


# haha = pd.read_csv('/Users/kunyue/project_personal/soccer/data/英超/soccer/game_end.csv')

# haha = haha[haha['away_team']=='切尔西'][['home_team','away_team','date','away_goal','home_goal']]

# hehe = pd.read_csv('/Users/kunyue/project_personal/soccer/data/英超/soccer/game_end.csv')
#
# hehe = hehe[(hehe['date']>='18/01/12') & (hehe['date']<'18/06/01')]
# haha = hehe[(hehe['away_team']=='斯托克城') | (hehe['home_team']=='斯托克城') ][['home_team','away_team','date','away_goal','home_goal']]
# print(haha)