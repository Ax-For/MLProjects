import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
import sklearn.metrics as sm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def plot_feature_importances(feature_importances, title, feature_names):
    feature_importances = 100 * (feature_importances / max(feature_importances))
    index_sorted = np.flipud


housing_data = datasets.load_boston()
X, y = shuffle(housing_data.data, housing_data.target, random_state=6)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

dt_reg = DecisionTreeRegressor(max_depth=4)
dt_reg.fit(x_train, y_train)

ad_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=6)
ad_reg.fit(x_train, y_train)

y_pred_dt = dt_reg.predict(x_test)
mse = sm.mean_squared_error(y_test, y_pred_dt)
evs = sm.explained_variance_score(y_test, y_pred_dt)
print("\n#### Decision Tree performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

y_pred_ad = ad_reg.predict(x_test)
mse = sm.mean_squared_error(y_test, y_pred_ad)
evs = sm.explained_variance_score(y_test, y_pred_ad)
print("\n#### Adaboost performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))