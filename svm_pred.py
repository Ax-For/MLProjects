import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC


def plot_classifier(classifier, input_data, target, title):
    """绘制分类器的分类边界"""
    x_min, x_max = input_data[:, 0].min() - 1, input_data[:, 0].max() + 1
    y_min, y_max = input_data[:, 1].min() - 1, input_data[:, 1].max() + 1

    step_size = 0.01
    x_val, y_val = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    mesh_out = classifier.predict(np.c_[x_val.ravel(), y_val.ravel()])
    mesh_out = mesh_out.reshape(x_val.shape)

    plt.figure()
    plt.pcolormesh(x_val, y_val, mesh_out, cmap=plt.cm.gray)

    plt.scatter(input_data[:, 0], input_data[:, 1], c=target, s=80, edgecolors="black", linewidth=1)

    plt.xlim(x_val.min(), x_val.max())
    plt.ylim(y_val.min(), y_val.max())

    plt.title(title)


if __name__ == '__main__':
    data = np.loadtxt("data/data_multivar_imbalance.txt", delimiter=",")
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=6)
    params = {"kernel": "linear", "class_weight": "balanced"}     # 线性核
    # params = {"kernel": "poly", "degree": 3}     # 多项式核
    # params = {"kernel": "rbf"}      # 径向基函数
    clf = SVC(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    plot_classifier(clf, X_test, y_test, "Testing dataset")
    # plot_classifier(clf, X_train, y_train, "Training dataset")
    plt.show()

    target_names = ['Class-' + str(i) for i in set(y)]
    print("\n" + "#" * 40)
    print("\nClassifier performance on testing dataset\n")
    print(classification_report(y_test, clf.predict(X_test), target_names=target_names))
    print("\n" + "#" * 40)

    """
    尝试修改并提交
    """

