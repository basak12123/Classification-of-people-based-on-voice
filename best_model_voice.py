import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from hmmlearn import hmm

rs = check_random_state(516)

def read_csv(file_name):
    return np.column_stack([np.loadtxt(file_name, delimiter=",", dtype=int, skiprows=1)[:, 1]])

def list_of_records(file_name):
    l = []
    for i in range(1, 10):
        l.append(read_csv("nagrania_csv/" + file_name + str(i) + "_Output_mono.csv"))
    return l

filip = list_of_records("filip/filip1_")
antoni = list_of_records("antoni/Nagranie")
jakub = list_of_records("jakub/kuba")
mateusz = list_of_records("mateusz/mateusz1_")
piotr = list_of_records("piotr/nagranie")

model1 = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100)
model1.fit(filip[1])
print(model1.score(filip[1]))

def model_aic_bic(X, rs, ns, iter = 200, name = ""):
    aic = []
    bic = []
    lls = []

    for n in ns:
        best_ll = None
        best_model = None
        for i in range(10):
            h = hmm.GaussianHMM(n_components=n, covariance_type="diag", n_iter=iter, random_state=rs)
            h.fit(X)
            score = h.score(X)
            if not best_ll or best_ll < score:
                best_ll = score
                best_model = h
        aic.append(best_model.aic(X))
        bic.append(best_model.bic(X))
        lls.append(best_model.score(X))

    fig, ax = plt.subplots()
    ln1 = ax.plot(ns, aic, label="AIC", color="blue", marker="o")
    ln2 = ax.plot(ns, bic, label="BIC", color="green", marker="o")
    ax2 = ax.twinx()
    ln3 = ax2.plot(ns, lls, label="LL", color="orange", marker="o")

    ax.legend(handles=ax.lines + ax2.lines)
    ax.set_title(f"Using AIC/BIC for Model Selection for {name}")
    ax.set_ylabel("Criterion Value (lower is better)")
    ax2.set_ylabel("LL (higher is better)")
    ax.set_xlabel("Number of HMM Components")
    fig.tight_layout()

    plt.show()
    print(aic, "\n", bic, "\n", lls)

ns = [9, 10, 11, 12]

model_aic_bic(filip[1], rs, ns, 200,"filip")