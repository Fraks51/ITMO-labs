import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
import numpy as np
from numpy.linalg import linalg
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge


def one_hot_encoder(d, columns_to_encode):
    return pd.get_dummies(d, columns=columns_to_encode, drop_first=True, dummy_na=True)


get_dummies_for = ['Parents',
                   'guardian',
                   'school_choice',
                   'time_to_home',
                   'school_time',
                   'spend_time_friends',
                   'school']


def h_ws(ws: np.ndarray, _xs: np.ndarray) -> float:
    return np.sum(ws[0:-1] * _xs) + ws[-1]


def find_part_rmse(_xs: np.ndarray, _ys: np.ndarray, model) -> float:
    vector_mul = model.predict(_xs)
    result = sum([(_ys[i] - vector_mul[i]) ** 2
                  for i in range(len(_ys))])
    return result


def get_x_ranking(x, y, num_of_columns):
    x = one_hot_encoder(x, get_dummies_for)
    estimator = Ridge(alpha=0.05, tol=0.0001)
    selector = RFE(estimator, n_features_to_select=num_of_columns, step=1)
    selector = selector.fit(x, y)
    support_map = dict(zip(x.columns, selector.support_))
    # test_x = pd.read_csv("../input/csc-hw3-autumn2020-team-4/2020_hw3_team_4_ds_test.csv").replace({
    #                                                 "yes": 1, "no": 0,
    #                                                 "T": 0, "A": 1, "m": 1,
    #                                                 "f": 0, "o": 2, "M": 1, "F": 0})
    # test_x = one_hot_encoder(test_x, get_dummies_for)
    # x = x.append(test_x)
    return x[filter(lambda feature: support_map[feature], x.columns)]


def get_rmse(new_x, y, tau, a, l1):
    result = 0
    kf = KFold(n_splits=15)
    for train_indexes, test_indexes in kf.split(new_x, y, ):
        x_test, x_train = new_x[test_indexes], new_x[train_indexes]
        y_test, y_train = y[test_indexes], y[train_indexes]
        ws = ElasticNet(alpha=a, l1_ratio=l1, tol=tau)
        ws.fit(x_train, y_train)
        result += find_part_rmse(x_test, y_test.to_numpy(), ws)
    return (result / len(y)) ** 0.5


def convert_data(data: str) -> int:
    args = list(map(int, data.split('-')))
    ans = (args[0] - 2000) * 365
    ans += args[1] * 30
    ans += args[2]
    return ans


def find_hyper():
    dataset = pd.read_csv("data/train1.csv")
    y = dataset["target"]
    x = dataset[["published", "last_update", "views", "recommendations", "comments", "size_kb", "rating", "category"]]
    x["published"] = x["published"].map(convert_data)
    x["last_update"] = x["last_update"].map(convert_data)
    x["div"] = x["last_update"] - x["published"]
    print(x)
    x = pd.get_dummies(x, prefix=['rating', 'category'])
    scaler = MinMaxScaler()
    scaler.fit(x)
    new_x = scaler.transform(x)
    min_loss, min_tau = 1000, 0
    for tau in [0.3, 0.5, 0.7, 0.9, 1]:
        for a in [0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]:
            for l1 in [0.7, 0.8, 0.85, 0.9]:
                loss = get_rmse(new_x, y, tau, a, l1)
                if min_loss > loss:
                    min_loss = loss
                print("<=====----=====>")
                print("tau: {}, alpha: {}, l1_ratio: {}".format(tau, a, l1))
                print("Loss: {}".format(loss))
    print("Best loss")
    print(min_loss)


def main():
    dataset = pd.read_csv("../input/csc-hw3-autumn2020-team-4/2020_hw3_team_4_ds_train_full.csv")
    dataset = dataset.replace({"yes": 1, "no": 0, "T": 0, "A": 1, "m": 1, "f": 0, "o": 2, "M": 1, "F": 0})
    y = dataset["target"]
    x = dataset.drop(["target"], axis=1)
    new_x = get_x_ranking(x, y, 9)
    scaler = MinMaxScaler()
    scaler.fit(new_x)
    new_x = scaler.transform(new_x)
    test_x = new_x[(len(y)):]
    train_x = new_x[:(len(y))]
    ws = mnk(train_x, y.to_numpy(), 0.369)
    ans = [h_ws(ws, i) for i in test_x]
    answer_df = pd.DataFrame()
    answer_df['id'] = pd.read_csv("../input/csc-hw3-autumn2020-team-4/2020_hw3_team_4_ds_test.csv")['id']
    answer_df['target'] = ans
    answer_df.to_csv("submisson.csv", index=False)


if __name__ == '__main__':
    find_hyper()
