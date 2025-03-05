import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.utils import check_random_state

rs = check_random_state(516)

def read_csv(file_name):
    return np.column_stack([np.loadtxt(file_name, delimiter=",", dtype=int, skiprows=1)[:, 1]])

def list_of_records(file_name):
    l = []
    for i in range(1, 11):
        l.append(read_csv("nagrania_csv/" + file_name + str(i) + "_Output_mono.csv"))
    return l

filip = list_of_records("filip/filip1_")
antoni = list_of_records("antoni/Nagranie")
jakub = list_of_records("jakub/kuba")
mateusz = list_of_records("mateusz/mateusz1_")
piotr = list_of_records("piotr/nagranie")
katarzyna = list_of_records("katarzyna/kas")

all_voices = [filip, antoni, jakub, mateusz, piotr, katarzyna]

def best_fit(dev):
    filip = list_of_records("filip/filip1_")
    antoni = list_of_records("antoni/Nagranie")
    jakub = list_of_records("jakub/kuba")
    mateusz = list_of_records("mateusz/mateusz1_")
    piotr = list_of_records("piotr/nagranie")
    katarzyna = list_of_records("katarzyna/kas")

    model_filip = GaussianHMM(n_components=10, covariance_type="diag", n_iter=2000, random_state=rs)
    model_antoni = GaussianHMM(n_components=10, covariance_type="diag", n_iter=2000, random_state=rs)
    model_jakub = GaussianHMM(n_components=10, covariance_type="diag", n_iter=2000, random_state=rs)
    model_mateusz = GaussianHMM(n_components=10, covariance_type="diag", n_iter=2000, random_state=rs)
    model_piotr = GaussianHMM(n_components=10, covariance_type="diag", n_iter=2000, random_state=rs)
    model_katarzyna = GaussianHMM(n_components=10, covariance_type="diag", n_iter=2000, random_state=rs)

    model_filip.fit(np.concatenate(filip[0:7]))
    print("1")
    model_antoni.fit(np.concatenate(antoni[0:7]))
    print("2")
    model_jakub.fit(np.concatenate(jakub[0:7]))
    print("3")
    model_mateusz.fit(np.concatenate(mateusz[0:7]))
    print("4")
    model_piotr.fit(np.concatenate(piotr[0:7]))
    print("5")
    model_katarzyna.fit(np.concatenate(katarzyna[0:7]))
    print("6")
    results = []
    for i in dev:
        res_ll = {"Filip": model_filip.score(i),
        "Antoni": model_antoni.score(i),
        "Jakub": model_jakub.score(i),
        "Mateusz": model_mateusz.score(i),
        "Piotr": model_piotr.score(i),
        "Katarzyna":model_katarzyna.score(i),}
        results.append(max(res_ll, key=res_ll.get))
    return results

print(best_fit(all_voices))
