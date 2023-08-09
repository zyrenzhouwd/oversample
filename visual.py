import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from smote_variants import MWMOTE
from over_cross import oversample_crossover
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from smgo import SMGO

f1, f2 = 0, 2


def plot_predictions(x_maj, x_min, x_os=None):
    # model = KNeighborsClassifier()
    model = SVC()
    if x_os is None:
        model.fit(np.concatenate([x_maj, x_min])[:, [f1, f2]], np.array([0 for i in range(x_maj.shape[0])] + [1 for i in range(x_min.shape[0])]))
    else:
        model.fit(np.concatenate([x_maj, x_min, x_os])[:, [f1, f2]], np.array([0 for i in range(x_maj.shape[0])] + [1 for i in range(x_min.shape[0] + x_os.shape[0])]))
    axes = plt.gca()
    x_1, x_2 = axes.get_xlim()
    y_1, y_2 = axes.get_ylim()
    x0s = np.linspace(x_1, x_2, int((x_2 - x_1) // 0.002))
    x1s = np.linspace(y_1, y_2, int((y_2 - y_1) // 0.002))
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = model.predict(X)
    y_pred = y_pred.reshape(x0.shape)
    # plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    if x_os is None:
        plt.contourf(x0, x1, y_pred, cmap=plt.cm.Blues, alpha=0.2)
    else:
        plt.contourf(x0, x1, y_pred, cmap=plt.cm.Blues, alpha=0.2)


def plot_data_all_test(x_maj, x_min, x_os, dataset=None, model=None):
    global f1, f2
    for f1 in range(0, x_maj.shape[1]):
        for f2 in range(f1 + 1, x_maj.shape[1]):
            plot_predictions(x_maj, x_min)
            plot_predictions(x_maj, x_min, x_os)
            plt.scatter(x_maj[:, f1], x_maj[:, f2], marker='o', c='blue', s=3, label='Majority')
            plt.scatter(x_min[:, f1], x_min[:, f2], marker='o', c='red', s=3, label='Minority')
            plt.scatter(x_os[:, f1], x_os[:, f2], marker='.', c='#00cb0a', s=3, label='Generated')
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.legend(loc='upper right')
            plt.savefig('./pics/{}_feat_{}+{}_{}.png'.format(dataset, f1, f2, model))
            plt.clf()


def plot_data(x_maj, x_min, x_os, dataset=None, model=None):
    global f1, f2
    # print(x_maj.shape[0], x_min.shape[0])
    plot_predictions(x_maj, x_min)
    plot_predictions(x_maj, x_min, x_os)
    plt.scatter(x_maj[:, f1], x_maj[:, f2], marker='o', c='blue', s=3, label='Majority')
    plt.scatter(x_min[:, f1], x_min[:, f2], marker='o', c='red', s=3, label='Minority')
    plt.scatter(x_os[:, f1], x_os[:, f2], marker='.', c='#00cb0a', s=3, label='Generated')
    # plt.tick_params(labelbottom=False, labelleft=False)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.legend(loc='upper right')
    plt.savefig('./pics/{}_feat_{}+{}_{}.svg'.format(dataset, f1, f2, model))
    plt.clf()


def oversampling(model_name, x, y, **kwargs):
    x_os, y_os = None, None
    if model_name == 'SMOTE':
        x_os, y_os = SMOTE(random_state=0).fit_resample(x, y)
    elif model_name == 'BorderlineSMOTE':
        x_os, y_os = BorderlineSMOTE(random_state=0).fit_resample(x, y)
    elif model_name == 'MWMOTE':
        x_os, y_os = MWMOTE(random_state=0).fit_resample(x, y)
    elif model_name == 'smgo':
        x_os, y_os = SMGO().fit_resample(x, y)
    return x_os[x.shape[0]:, :], y_os[x.shape[0]:]


if __name__ == '__main__':
    from data_process import process_yeast, process_magic, process_shuttle

    maj_limit, min_limit = 400, 80
    x_maj, y_maj, x_min, y_min = process_yeast()
    x_maj, y_maj, x_min, y_min = x_maj[:maj_limit, :], y_maj[:maj_limit], x_min[:min_limit, :], y_min[:min_limit]

    x_os, y_os = oversampling('SMOTE', np.concatenate([x_maj, x_min], axis=0), np.concatenate([y_maj, y_min], axis=0))
    # plot_predictions(x_maj, x_min, x_os)
    plot_data(x_maj, x_min, x_os, 'yeast', 'SMOTE')
    # plot_data_all_test(x_maj, x_min, x_os, 'yeast', 'SMOTE')

    x_os, y_os = oversampling('BorderlineSMOTE', np.concatenate([x_maj, x_min], axis=0), np.concatenate([y_maj, y_min], axis=0))
    # plot_predictions(x_maj, x_min, x_os)
    plot_data(x_maj, x_min, x_os, 'yeast', 'BorderlineSMOTE')
    # plot_data_all_test(x_maj, x_min, x_os, 'yeast', 'BorderlineSMOTE')

    x_os, y_os = oversampling('MWMOTE', np.concatenate([x_maj, x_min], axis=0), np.concatenate([y_maj, y_min], axis=0))
    # plot_predictions(x_maj, x_min, x_os)
    plot_data(x_maj, x_min, x_os, 'yeast', 'MWMOTE')
    # plot_data_all_test(x_maj, x_min, x_os, 'yeast', 'MWMOTE')

    x_os, y_os = oversampling('smgo', np.concatenate([x_maj, x_min], axis=0), np.concatenate([y_maj, y_min], axis=0), row_1=400)
    # plot_predictions(x_maj, x_min, x_os)
    # plot_data(x_maj, x_min, x_os, 'yeast', 'crossover')
    # plot_data_all_test(x_maj, x_min, x_os, 'yeast', 'smgo')
    plot_data(x_maj, x_min, x_os, 'yeast', 'smgo')
