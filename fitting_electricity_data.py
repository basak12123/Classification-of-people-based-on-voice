import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.utils import check_random_state

rs = check_random_state(516)

dev1 = np.column_stack(
        [np.loadtxt("python_public/test_folder2/dev1.csv", delimiter=",", dtype=int, skiprows=1)[:, 1]])
dev2 = np.column_stack(
        [np.loadtxt("python_public/test_folder2/dev2.csv", delimiter=",", dtype=int, skiprows=1)[:, 1]])
dev3 = np.column_stack(
        [np.loadtxt("python_public/test_folder2/dev3.csv", delimiter=",", dtype=int, skiprows=1)[:, 1]])
dev4 = np.column_stack(
        [np.loadtxt("python_public/test_folder2/dev4.csv", delimiter=",", dtype=int, skiprows=1)[:, 1]])
dev5 = np.column_stack(
        [np.loadtxt("python_public/test_folder2/dev5.csv", delimiter=",", dtype=int, skiprows=1)[:, 1]])
dev6 = np.column_stack(
        [np.loadtxt("python_public/test_folder2/dev6.csv", delimiter=",", dtype=int, skiprows=1)[:, 1]])

devs = [dev1, dev2, dev3, dev4, dev5, dev6]

def best_fit(dev):
    arr = np.column_stack(
        [np.loadtxt("python_public/house3_5devices_train.csv", delimiter=",", dtype=int, skiprows=1)])

    lighting2 = np.column_stack([arr[:, 1]])
    lighting5 = np.column_stack([arr[:, 2]])
    lighting4 = np.column_stack([arr[:, 3]])
    refrigerator = np.column_stack([arr[:, 4]])
    microwave = np.column_stack([arr[:, 5]])

    model_lighting2 = GaussianHMM(n_components=9, covariance_type="diag", n_iter=200, random_state=rs)
    model_lighting5 = GaussianHMM(n_components=10, covariance_type="diag", n_iter=200, random_state=rs)
    model_lighting4 = GaussianHMM(n_components=9, covariance_type="diag", n_iter=200, random_state=rs)
    model_refrigerator = GaussianHMM(n_components=11, covariance_type="diag", n_iter=200, random_state=rs)
    model_microwave = GaussianHMM(n_components=10, covariance_type="diag", n_iter=200, random_state=rs)

    model_lighting2.fit(lighting2)
    model_lighting5.fit(lighting5)
    model_lighting4.fit(lighting4)
    model_refrigerator.fit(refrigerator)
    model_microwave.fit(microwave)
    results = []
    for i in dev:
        res_ll = {"lighting2": model_lighting2.score(i),
        "lighting5": model_lighting5.score(i),
        "lighting4": model_lighting4.score(i),
        "refrigerator": model_refrigerator.score(i),
        "microwave": model_microwave.score(i)}
        results.append(max(res_ll, key=res_ll.get))
    return results

print(best_fit(devs))
