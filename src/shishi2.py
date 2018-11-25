import pandas as pd


p1 = {'home_team':['埃弗顿','埃弗顿','埃弗顿'],'away_team':['热刺','曼联','沃特福德'],'date':['17/10/14','17/11/01','17/11/06'],'home_goal':[2,3,1]}
p2 = {'home_team':['埃弗顿','埃弗顿'],'away_team':['曼联','沃特福德'],'date_index':['17/11/01','17/11/06']}


index = pd.DataFrame(p2)
match = pd.DataFrame(p1)


index