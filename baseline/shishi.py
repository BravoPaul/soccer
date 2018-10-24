import os
import pandas as pd

p = 1

x = 'dffas'

y = 'fasdf'


print(x+str(p)+y)

x = x.replace('away','temp_away_home')
x = x.replace('home','away')
x = x.replace('temp_away_home','home')