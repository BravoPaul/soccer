import os
import pandas as pd

import numpy as np
import pandas as pd
pd.options.display.width = 1000

df = pd.DataFrame(
    {'home_team': ['yy','kk','yy','yy'],
     'away_team': ['bb','cc','bb','bb'],
     'date': [2,1,3,5]})


df.columns = sorted(df.columns)

df2 = pd.DataFrame(
    {'away_team': ['yy','kk','yy','yy'],
     'home_team': ['bb','cc','bb','bb'],
     'date': [2,1,3,5]})

df2.columns = sorted(df2.columns)



print(pd.concat([df,df2],axis=0))
