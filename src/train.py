from sklearn_pandas import DataFrameMapper
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA



def train(train_data,label):

    match = pd.read_csv('/Users/kunyue/project_personal/soccer/data/英超/soccer/game_end.csv')

    def main_bet_predictor(x):
        if (x['main_bet']<0) and (x['home_goal']>x['away_goal']):
            return 1
        elif (x['main_bet']>0) and (x['home_goal']<x['away_goal']):
            return -1
        elif (x['main_bet'] == 0) and (x['home_goal'] == x['away_goal']):
            return 1
        return 0
    y_bet = match.apply(lambda x: main_bet_predictor(x),axis=1).values
    print("准确率至少为： ", np.count_nonzero(y_bet)/len(y_bet))


    # 多分类
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    for train_index, test_index in kf.split(train_data):
        xgboost_model = xgb.XGBClassifier().fit(train_data[train_index], label[train_index])
        # xgb.XGBRegressor
        # 预测结果
        pred = xgboost_model.predict(train_data[test_index])
        # 标准答案
        ground_truth = label[test_index]
    print("混淆矩阵:")
    print(confusion_matrix(ground_truth, pred))
    print("准确率：")
    print(metrics.accuracy_score(ground_truth, pred))
6



if __name__ == "__main__":
    pd_train_data = pd.read_csv('/Users/kunyue/project_personal/soccer/data/train_data/train_data.csv')
    pd_label_data = pd_train_data['win_lose_label']
    del pd_train_data['home_team']
    del pd_train_data['away_team']
    del pd_train_data['date']
    del pd_train_data['win_lose_label']
    pd_train_data = pd_train_data.fillna(0)
    train(pd_train_data.values,pd_label_data.values)

