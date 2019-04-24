import numpy as np
import gist
import cv2
import os
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
    for r_dir, s_dir, files in os.walk("./data_align"):
        if len(s_dir) > 0:
            continue
        s_target = r_dir.split("/")[-1][1:]
        for file in files:
            path_file = os.path.join(r_dir, file)
            s_feature = gist.extract(cv2.imread(path_file))
            data.append(np.append(s_feature, int(s_target)))
    data = np.asarray(data)
    data = data[data[:, -1].argsort()]
    feature = data[:, :-1]
    target = data[:, -1].astype(int)

    # SVC
    model_svc = SVC()

    cv = custom_cv_5folds()

    scores_svc = cross_val_score(model_svc, feature, target, cv=cv)

    print("SVC: {}".format(scores_svc), scores_svc.mean())

    # KNN
    model_knn = KNeighborsClassifier(n_neighbors=5, p=2, weights="distance")

    cv = custom_cv_5folds()

    scores_knn = cross_val_score(model_knn, feature, target, cv=cv)

    print("KNN: {}".format(scores_knn), scores_knn.mean())
