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


# haha = pd.read_csv('/Users/kunyue/project_personal/soccer/data/split_data/data_17_17-08-12_17-11-24_data.csv')
#
# print(haha[haha['home_team']=='曼城'][['away_goal','home_team','away_team']])
#
# hehe = pd.read_csv('/Users/kunyue/project_personal/soccer/data/英超/soccer/game_end.csv')
#
# hehe = hehe[(hehe['date']>='17/08/12') & (hehe['date']<'17/11/24')]
#
# print(hehe[(hehe['home_team']=='曼城') | (hehe['away_team']=='曼城')][['home_goal','home_team','away_team','away_goal','date']])