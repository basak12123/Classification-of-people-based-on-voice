import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from hmmlearn import hmm

rs = check_random_state(516)

arr = np.loadtxt("python_public/house3_5devices_train.csv", delimiter=",", dtype=int, skiprows=1)
test_data = np.loadtxt("python_public/test_folder2/dev1.csv", delimiter=",", dtype=int, skiprows=1)

lighting2 = np.column_stack([arr[:, 1]])
lighting5 = np.column_stack([arr[:, 2]])
lighting4 = np.column_stack([arr[:, 3]])
refrigerator = np.column_stack([arr[:, 4]])
microwave = np.column_stack([arr[:, 5]])

model1 = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100)
model1.fit(lighting2)

hidden_state = model1.predict(lighting2)

for i in range(model1.n_components):
    print('Hidden state', i+1)
    print('Mean =', round(model1.means_[i][0], 3))
    print('Variance =', round(np.diag(model1.covars_[i])[0], 3))

lengths = len(lighting2)
ns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

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


model_aic_bic(lighting2, rs, ns, 200,"lighting2")
model_aic_bic(lighting4, rs, ns, 200,"lighting4")
model_aic_bic(lighting5, rs, ns, 200,"lighting5")
model_aic_bic(microwave, rs, ns, 200,"microwave")
model_aic_bic(refrigerator, rs, ns, 200,"refrigerator")
