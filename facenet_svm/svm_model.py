import os
import numpy as np
from scipy.special._ufuncs import ker
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit


def custom_cv_5folds():
    for step in range(5):
        idx = set()
        for i in range(40):
            pos = np.add(np.add(np.array([0, 1]), 2 * step), 10 * i)
            for val in pos:
                idx.add(val)
        result = np.asarray(list(idx), dtype=int)
        yield result, result


if __name__ == "__main__":

    # Load dataset
    data = list()
    for r_dir, s_dir, files in os.walk("./embedding"):
        if len(s_dir) > 0:
            continue
        s_target = r_dir.split("/")[-1][1:]
        for file in files:
            s_feature = np.load(os.path.join(r_dir, file))[0]
            data.append(np.append(s_feature, int(s_target)))
    data = np.asarray(data)
    data = data[data[:, -1].argsort()]
    feature = data[:, :-1]
    target = data[:, -1].astype(int)

    # Test SVC
    train_idx = set(range(400))
    test_idx = set()
    for i in range(40):
        pos = np.add(np.array([8, 9]), 10 * i)
        for val in pos:
            test_idx.add(val)
    train_idx -= test_idx
    feature_train = feature[list(train_idx), :]
    feature_test = feature[list(test_idx), :]
    target_train = target[list(train_idx)]
    target_test = target[list(test_idx)]

    model_svc = SVC(C=1.0, kernel="rbf", gamma=0.001, probability=True)

    model_svc.fit(feature_train, target_train)

    print("SVC: {}".format(model_svc.score(feature_test, target_test)))

    model_knn = KNeighborsClassifier(n_neighbors=5, p=2, weights="distance")

    model_knn.fit(feature_train, target_train)

    print("KNN: {}".format(model_knn.score(feature_test, target_test)))

    print(model_svc.predict_proba(feature_test)[2])
    print(model_svc.decision_function(feature_test)[2])

    # SVC
    model_svc = SVC(C=1.0, kernel="rbf", gamma=0.001, probability=True)

    cv = custom_cv_5folds()

    scores_svc = cross_val_score(model_svc, feature, target, cv=cv)

    print("SVC: {}".format(scores_svc))

    # KNN
    model_knn = KNeighborsClassifier(n_neighbors=5, p=2, weights="distance")

    cv = custom_cv_5folds()

    scores_knn = cross_val_score(model_knn, feature, target, cv=cv)

    print("KNN: {}".format(scores_knn))
