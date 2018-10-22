#!/usr/bin/env python
# coding: utf-8
# File: analyse.py
# Author: BravoPaul


import pandas as pd


date = []
home_red = []
home_yellow = []
home_team = []
home_goal = []
away_goal = []
away_team = []
away_yellow = []
away_red = []
home_goal_half = []
away_goal_half = []
home_bet = []
goals_bet = []
footGoal_bet = []
footGoal_home_half = []
footGoal_away_half = []
footGoal_home = []
footGoal_away = []

f = open("/Users/kunyue/project_personal/soccer/data/巴西保杯", "r")
lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.replace('-', '-0')
    line_list = line.split('\t')
    if len(line_list) not in [11,12]:
        print(len(line))
    if len(line_list)==11:
        line_list.insert(1," ")
    elif len(line_list)<11:
        continue
    else:
        date.append(line_list[2])
        if len(line_list[3].split(' '))>1:
            try:
                yellow_red_int = int(line_list[3].split(' ')[0])
                home_team.append(''.join(line_list[3].split(' ')[1:]))
            except ValueError:
                yellow_red_int = 0
                home_team.append(''.join(line_list[3].split(' ')[0:]))
            home_red.append(int(yellow_red_int / 10 % 10))
            home_yellow.append(int(yellow_red_int % 10))
        else:
            home_team.append(line_list[3])
            home_yellow.append(0)
            home_red.append(0)
        home_goal.append(line_list[4].split(' ')[0])
        away_goal.append(line_list[4].split(' ')[2])
        if len(line_list[5].split(' '))>1:
            try:
                yellow_red_int = int(line_list[5].split(' ')[-1])
                away_team.append(''.join(line_list[5].split(' ')[0:-1]))
            except:
                yellow_red_int = 0
                away_team.append(''.join(line_list[5].split(' ')[0:]))
            away_red.append(int(yellow_red_int / 10 % 10))
            away_yellow.append(int(yellow_red_int % 10))
        else:
            away_team.append(line_list[5])
            away_yellow.append(0)
            away_red.append(0)
        home_goal_half.append(line_list[6].split(' ')[0])
        away_goal_half.append(line_list[6].split(' ')[2])
        home_bet.append(float(line_list[7].split(' ')[0]))
        goals_bet.append(float(line_list[7].split(' ')[2]))
        footGoal_bet.append(float(line_list[7].split(' ')[4]))
        footGoal_home_half.append(float(line_list[9].split(' ')[0]))
        footGoal_away_half.append(float(line_list[9].split(' ')[2]))
        footGoal_home.append(float(line_list[10].split(' ')[0]))
        footGoal_away.append(float(line_list[10].split(' ')[2]))

data = {
    'date': date,
    'home_red': home_red,
    'home_yellow': home_yellow,
    'home_team': home_team,
    'home_goal': home_goal,
    'away_goal': away_goal,
    'away_team': away_team,
    'away_yellow': away_yellow,
    'away_red': away_red,
    'home_goal_half': home_goal_half,
    'away_goal_half': away_goal_half,
    'home_bet': home_bet,
    'goals_bet': goals_bet,
    'footGoal_bet': footGoal_bet,
    'footGoal_home_half': footGoal_home_half,
    'footGoal_away_half': footGoal_away_half,
    'footGoal_home': footGoal_home,
    'footGoal_away': footGoal_away,
}

df = pd.DataFrame(data)
df.to_csv('/Users/kunyue/project_personal/soccer/data/output/巴西保杯.csv')

