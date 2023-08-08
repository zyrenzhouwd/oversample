import pandas as pd
import numpy as np


def process_page_block():
    df = pd.read_csv('./data/page_blocks/page-blocks.data', delimiter='\s+', header=None)

    df_maj = df[df[10] == 2]
    df_min = df[df[10] == 5]

    x_maj, y_maj = df_maj.iloc[:, :10].to_numpy(dtype=np.float64), np.array([0 for i in range(df_maj.shape[0])])
    x_min, y_min = df_min.iloc[:, :10].to_numpy(dtype=np.float64), np.array([1 for i in range(df_min.shape[0])])

    return x_maj, y_maj, x_min, y_min


def process_yeast():
    df = pd.read_csv('./data/yeast/yeast.data', delimiter='\s+', header=None)

    df_maj = df[df[9] == 'CYT']
    df_min = df[df[9] == 'ME3']

    x_maj, y_maj = df_maj.iloc[:, 1:9].to_numpy(dtype=np.float64), np.array([0 for i in range(df_maj.shape[0])])
    x_min, y_min = df_min.iloc[:, 1:9].to_numpy(dtype=np.float64), np.array([1 for i in range(df_min.shape[0])])

    return x_maj, y_maj, x_min, y_min


def process_shuttle():
    df = pd.read_csv('./data/statlog+shuttle/shuttle.csv', delimiter='\s+', header=None)

    df_maj = df[df[9] == 1]
    df_min = df[df[9] == 4]

    print('maj_num: {}, min_num: {}'.format(df_maj.shape[0], df_min.shape[0]))

    x_maj, y_maj = df_maj.iloc[:, :9].to_numpy(dtype=np.float64), np.array([0 for i in range(df_maj.shape[0])])
    x_min, y_min = df_min.iloc[:, :9].to_numpy(dtype=np.float64), np.array([1 for i in range(df_min.shape[0])])

    return x_maj, y_maj, x_min, y_min

def process_magic():
    df = pd.read_csv('./data/magic+gamma+telescope/magic04.data', delimiter=',', header=None)

    df_maj = df[df[10] == 'g']
    df_min = df[df[10] == 'h']

    print('maj_num: {}, min_num: {}'.format(df_maj.shape[0], df_min.shape[0]))

    x_maj, y_maj = df_maj.iloc[:, :-1].to_numpy(dtype=np.float64), np.array([0 for i in range(df_maj.shape[0])])
    x_min, y_min = df_min.iloc[:, :-1].to_numpy(dtype=np.float64), np.array([1 for i in range(df_min.shape[0])])

    return x_maj, y_maj, x_min, y_min
