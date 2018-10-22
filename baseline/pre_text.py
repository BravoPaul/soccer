#!/usr/bin/env python
# coding: utf-8
# File: analyse.py
# Author: BravoPaul


import pandas as pd


date = []                   #比赛日期
home_red = []               #主队红牌
home_yellow = []            #主队黄牌
home_team = []              #主队队名
home_goal = []              #主队进球
away_goal = []              #客队进球
away_team = []              #客队队名
away_yellow = []            #客队黄牌
away_red = []               #客队红牌
home_goal_half = []         #主队半场进球
away_goal_half = []         #客队半场进球
home_bet = []               #主队让球个数
goals_bet = []              #全场博彩大小球
footGoal_bet = []           #全场角球博彩大小球
footGoal_home_half = []     #主队半场角球个数
footGoal_away_half = []     #客队半场角球个数
footGoal_home = []          #主队全场角球个数
footGoal_away = []          #客队全场角球个数

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

