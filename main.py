import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score


def load_dataset(x: str, y: str):
    features = np.load(x).astype(np.float32)
    labels = np.load(y).astype(np.int)
    labels = np.argmax(labels, axis=1)
    return features, labels


def load_testing_set(x: str):
    def encode(char: str):
        if char == 'A':
            return 0
        elif char == 'T':
            return 1
        elif char == 'G':
            return 2
        elif char == 'C':
            return 3

    dataframe = pd.read_csv(x)
    ids = dataframe['ID'].to_numpy()
    features = dataframe['reads'].to_numpy()
    encoded = []

    for column in features:
        encoded.append([encode(i) for i in column])

    return ids.astype(np.int), np.array(encoded).astype(np.float32)


if __name__ == '__main__':
    # initialize the dataset first.
    x_train, y_train = load_dataset(x='train_x.npy', y='train_y.npy')
    x_train_0, y_train_0 = x_train[np.where(y_train == 0)[0]], y_train[np.where(y_train == 0)[0]]
    x_train_1, y_train_1 = x_train[np.where(y_train == 1)[0]], y_train[np.where(y_train == 1)[0]]
    # randomly sample some data to make the dataset looks more equally distributed.
    x_train_1, y_train_1 = sklearn.utils.shuffle(x_train_1, y_train_1, random_state=1)
    x_train_1, y_train_1 = x_train_1[:(19713*2), :], y_train_1[:(19713*2)]
    x_train_2, y_train_2 = x_train[np.where(y_train == 2)[0]], y_train[np.where(y_train == 2)[0]]
    # randomly sample some data to make the dataset looks more equally distributed.
    x_train_2, y_train_2 = sklearn.utils.shuffle(x_train_2, y_train_2, random_state=1)
    x_train_2, y_train_2 = x_train_2[:(19713*2), :], y_train_2[:(19713*2)]
    x_train_lite = np.concatenate([x_train_0, x_train_1, x_train_2], axis=0)
    y_train_lite = np.concatenate([y_train_0, y_train_1, y_train_2], axis=0)
    cf = RandomForestClassifier(random_state=1, n_jobs=4, verbose=10, n_estimators=200,
                                oob_score=True).fit(x_train_lite, y_train_lite)
    # cf = MLPClassifier(hidden_layer_sizes=(256, 256, 128, 96, 96, 64, 64), learning_rate_init=3e-04,
    #                    batch_size=20480, max_iter=100, random_state=114, learning_rate='adaptive',
    #                    verbose=1).fit(x_train, y_train)
    # class_weight={0: 9, 1: 0.6, 2: 0.4}
    # print(cf.oob_score_)
    # print(cf.best_loss_)
    # print(cf.loss_)
    x_val, y_val = load_dataset(x='val_x.npy', y='val_y.npy')
    # res_train = cf.score(x_train, y_train)
    # print(res_train)
    # res_val_0 = cf.score(x_val[np.where(y_val == 0)[0]], y_val[np.where(y_val == 0)[0]])
    # res_val_1 = cf.score(x_val[np.where(y_val == 1)[0]], y_val[np.where(y_val == 1)[0]])
    # res_val_2 = cf.score(x_val[np.where(y_val == 2)[0]], y_val[np.where(y_val == 2)[0]])
    # print("ACC of Class 0: {}".format(res_val_0))
    # print("ACC of Class 1: {}".format(res_val_1))
    # print("ACC of Class 2: {}".format(res_val_2))
    res_val = cf.predict(x_val)
    macro_f1 = f1_score(res_val, y_val, average='macro')
    print("Mean F1 Score {}".format(macro_f1))
    # ids, x_test = load_testing_set('./test.csv')
    # res_test = cf.predict(x_test)
    # res = {'ID': ids, 'label': res_test}
    # df = pd.DataFrame(data=res, dtype=np.int)
    # df.to_csv('./output_{}_lite.csv'.format(cf.__class__.__name__), index=False)
