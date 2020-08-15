from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    input_data_file = 'data/car.data.csv'
    df = pd.read_csv(input_data_file, header=None)
    x = df.values
    label_encoded = []
    x_encoded = np.empty(x.shape)
    for i, item in enumerate(x[0]):
        label_encoded.append(preprocessing.LabelEncoder())
        x_encoded[:, i] = label_encoded[i].fit_transform(x[:, i])
    x = x_encoded[:, :-1].astype(int)
    y = x_encoded[:, -1].astype(int)

    params = {"n_estimators": 200, "max_depth": 8, "random_state": 6}
    clf = RandomForestClassifier(**params)
    clf.fit(x, y)
    model_selection.cross_val_score(clf, x, y)