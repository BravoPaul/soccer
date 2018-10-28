import os
import pandas as pd

import numpy as np
import pandas as pd
pd.options.display.width = 1000

df = pd.DataFrame(
    {'home_team': ['yy','kk','yy','yy'],
     'away_team': ['bb','cc','bb','bb'],
     'date': [2,1,3,5]})

grouped = df.groupby(['home_team','away_team'])['date']
df['second_lowest'] = grouped.transform(lambda x: x.nsmallest(2).max())
df['has_null'] = grouped.transform(lambda x: pd.isnull(x).any()).astype(bool)
print(df)

pd_all = pd.read_csv("../data/英超/英超.csv")
pd_goal = pd_all.sort_values(by=['home_team','away_team','date'],ascending=False)
grp = pd_goal.groupby(['home_team','away_team']).nth(1).reset_index()
print(grp[['home_team','away_team','date']])