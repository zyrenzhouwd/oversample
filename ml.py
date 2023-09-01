from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from smote_variants import MWMOTE
import logging
import smote_variants as sv
from over_cross import oversample_crossover
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from smgo import SMGO

logging.getLogger(sv.__name__).setLevel(logging.CRITICAL)


def oversampling(model_name, x, y, **kwargs):
    x_os, y_os = None, None
    if model_name == 'SMOTE':
        x_os, y_os = SMOTE(sampling_strategy=kwargs['sampling_strategy'], random_state=0).fit_resample(x, y)
    elif model_name == 'BorderlineSMOTE':
        x_os, y_os = BorderlineSMOTE(sampling_strategy=kwargs['sampling_strategy'], random_state=0).fit_resample(x, y)
    elif model_name == 'MWMOTE':
        x_os, y_os = MWMOTE(proportion=kwargs['proportion'], random_state=0, n_jobs=4).fit_resample(x, y)
    elif model_name == 'smgo':
        x_os, y_os = SMGO(r=kwargs['r']).fit_resample(x, y)
    return x_os[x.shape[0]:, :], y_os[x.shape[0]:]


def get_metric(output, y_true):
    # print(classification_report(y_true, output))
    return precision_recall_fscore_support(y_true, output)


def dataset_split(x, y):
    # 使用 random_split 函数进行数据集分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    from data_process import process_magic
    import os
    import zipfile
    import datetime
    from datetime import datetime

    print('>>> Processing Data...')
    x_train, x_test, y_train, y_test = None, None, None, None
    if os.path.exists('./x_train.npy') and os.path.exists('./x_test.npy') and os.path.exists('./y_train.npy') and os.path.exists('./y_test.npy'):
        x_train, x_test, y_train, y_test = np.load('./x_train.npy'), np.load('./x_test.npy'), np.load('./y_train.npy'), np.load('./y_test.npy')
    else:
        maj_limit, min_limit = 12000, 1000
        x_maj, y_maj, x_min, y_min = process_magic()
        x_maj, y_maj, x_min, y_min = x_maj[:maj_limit, :], y_maj[:maj_limit], x_min[:min_limit, :], y_min[:min_limit]
        x_train, x_test, y_train, y_test = dataset_split(np.concatenate([x_maj, x_min]), np.concatenate([y_maj, y_min]))
        np.save('./x_train', x_train)
        np.save('./x_test', x_test)
        np.save('./y_train', y_train)
        np.save('./y_test', y_test)

    print('>>> Oversampling...')

    model = None
    params = {'b': 0, 'w': 1, 'k1':3, 'k2': 2, 'k3': 20}
    for method in ['Original', 'SMOTE', 'BorderlineSMOTE', 'MWMOTE', 'smgo']:
    # for method in ['smgo']:
        print('>>> Method: {:20s}'.format(method))
        result = {}
        for r in range(1, 11):
            x_os, y_os = None, None
            x_train_new, y_train_new = x_train, y_train
            if method == 'smgo':
                x_os, y_os = oversampling('smgo', x_train_new, y_train_new, b=params['b'], w=params['w'], k1=params['k1'], k2=params['k2'], k3=params['k3'], r=(r + 1) * y_train_new.sum() / (y_train_new.shape[0] - y_train_new.sum()))
            elif method == 'MWMOTE':
                x_os, y_os = oversampling('MWMOTE', x_train_new, y_train_new, proportion=(r + 1) * y_train_new.sum() / (y_train_new.shape[0] - y_train_new.sum()))
            elif method == 'SMOTE' or method == 'BorderlineSMOTE':
                x_os, y_os = oversampling(method, x_train_new, y_train_new, sampling_strategy=(r + 1) * y_train_new.sum() / (y_train_new.shape[0] - y_train_new.sum()))

            if method != 'Original':
                x_train_new, y_train_new = np.concatenate([x_train_new, x_os], axis=0), np.concatenate([y_train_new, y_os], axis=0)
            elif r == 2:
                break
            for model in [RandomForestClassifier(random_state=0), xgb.XGBClassifier(random_state=0), MLPClassifier([16, 8, 4], random_state=0, max_iter=10000, learning_rate_init=0.01)]:
            # for model in [RandomForestClassifier(random_state=0)]:
                print('    Model: {}'.format(model.__class__.__name__))
                model.fit(x_train_new, y_train_new)
                pred = model.predict(x_test)

                rate = 0 if x_os is None else r
                print('      Rate: {}'.format(rate))
                scores = get_metric(pred, y_test)
                # print('      Class Maj: Precision: {:.5f}  Recall: {:.5f}  F1: {:.5f}'.format(scores[0][0], scores[1][0], scores[2][0]))
                print('        Class Min: Precision: {:.5f}  Recall: {:.5f}  F1: {:.5f}'.format(scores[0][1], scores[1][1], scores[2][1]))
                if model.__class__.__name__ not in result:
                    result[model.__class__.__name__] = []
                result[model.__class__.__name__].append([rate, scores[1][1], scores[2][1]])
        for n in result:
            pd.DataFrame(result[n]).to_csv(f'./metric/{n}_{method}.csv', header=None, index=False)
    # with zipfile.ZipFile('metric_{}.zip'.format('-'.join([str(params[k]) for k in params])), 'w') as myzip:
    #     myzip.write('./metric')
