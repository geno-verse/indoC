#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, matplotlib.pyplot as plt, numpy as np, pandas as pd
import seaborn as sns

sns.set_theme()
sns.set_style("white")
from glob import glob
from common_functions import load_data

from scipy import stats

import time
from trace_extract_funcs import (
    get_nn,
    seq_dict,
    col_list,
    categorise_pivot,
    search_multiPos,
    search_multiPos_link,
    dict_filter,
    per_pos_feature,
    per_pos_feature_list,
)

start = time.time()


# Load data
nn = 4  # how many neighbour positions to take into account
dt_shift = 10  # expected shift between center of the pore and motor protein in bases
features = ["si", "tr", "dt0", "dt%s" % dt_shift]


fasta = "yeast_rRNA.fa"  # reference FastA
fnpat_noncmc = "*.bam"  # pattern for all BAM files
fnpat_cmc = "*.bam"  # pattern for all BAM files

modpos = pd.read_csv(
    "RNA_Mod_Positions_rRNAYeast.tsv",
    sep="\t",
)
all_pos_mod = modpos.query('Mod!="Unm"')
all_pos_mod = all_pos_mod["Chr_Position"].to_list()


# positions of all Y modifications
Ymodpos = modpos.query('Mod=="Y"')
Ymodpos = Ymodpos[["Chr_Position"]]
Y_pos_mod = Ymodpos["Chr_Position"].tolist()


cov_cut = 50
mis_cut = 0.05

# Valid U sites - noncmc
sites_noncmc = pd.read_csv("yeast_rRNA.per.site.baseFreq.csv")
sites_noncmc["Chr_Position"] = (
    sites_noncmc["#Ref"] + str("_") + sites_noncmc["pos"].astype(str)
)
sites_noncmc.rename({"#Ref": "Chr", "pos": "Position"}, axis=1, inplace=True)
sites_noncmc = sites_noncmc[["Chr_Position", "base", "cov", "mis"]]
chr_pos_sites_noncmc = sites_noncmc.query(
    'cov>{} and base=="T" and mis>={}'.format(cov_cut, mis_cut)
)
chr_pos_sites_noncmc = chr_pos_sites_noncmc["Chr_Position"].tolist()

# Valid U sites - cmc
sites_cmc = pd.read_csv("yeast-cmc.per.site.baseFreq.csv")
sites_cmc["Chr_Position"] = sites_cmc["#Ref"] + str("_") + sites_cmc["pos"].astype(str)
sites_cmc.rename({"#Ref": "Chr", "pos": "Position"}, axis=1, inplace=True)
sites_cmc = sites_cmc[["Chr_Position", "base", "cov", "mis"]]
chr_pos_sites_cmc = sites_cmc.query(
    'cov>{} and base=="T" and mis>={}'.format(cov_cut, mis_cut)
)
chr_pos_sites_cmc = chr_pos_sites_cmc["Chr_Position"].tolist()

common_sites = list(set(chr_pos_sites_cmc).intersection(set(chr_pos_sites_noncmc)))
common_sites = list(set(common_sites).difference(all_pos_mod))
HE_pos_mod = common_sites


def ks_extract(feat, Y_pos_mod):

    Y_sites_dict = {}
    for modpos in Y_pos_mod:
        Y_sites_dict[modpos] = [modpos]

    bams_noncmc = sorted(glob(fnpat_noncmc))
    bams_cmc = sorted(glob(fnpat_cmc))

    def get_regions(chr_pos):
        regions = [(cp.split("_")[0], int(cp.split("_")[-1]), "_") for cp in chr_pos]
        return regions

    def trace_df(fasta, bams, regions, features):

        samples = [fn.split(os.path.sep)[-3].split("_")[-1] for fn in bams]
        print(samples, regions)
        region2data = load_data(fasta, bams, regions, features, nn=nn)

        # define features
        feature_names = [
            "%s_%s" % (f.upper(), i) for f in features for i in range(-nn, nn + 1)
        ]
        len(feature_names), len(region2data)

        # concatenate all pos and samples into one dataframe
        dframes = []
        for ri, (ref, pos) in enumerate(
            region2data.keys()
        ):  # regions): #[3]#; print(ref, pos, mt)
            mer, calls = region2data[(ref, pos)]
            for c, s in zip(calls, samples):
                df = pd.DataFrame(c, columns=feature_names)
                df["Strain"] = s
                df["chr_pos"] = "%s_%s" % (ref, pos)
                dframes.append(df)
        # read all tsv files
        df = pd.concat(dframes).dropna().reset_index()
        df.head()
        return df

    mod_regions = get_regions(Y_pos_mod)

    # df with TR and SI of all Y mods - noncmc
    df_Y_noncmc = trace_df(fasta, bams_noncmc, mod_regions, features)
    # df with TR and SI of all Y mods - noncmc
    df_Y_cmc = trace_df(fasta, bams_cmc, mod_regions, features)

    # remove all columns except Trace and Signal intensity
    pos_col_list = col_list(nn)
    pos_col_list.append("chr_pos")
    Y_trace_pos_noncmc = df_Y_noncmc[df_Y_noncmc.columns.intersection(pos_col_list)]
    Y_trace_pos_cmc = df_Y_cmc[df_Y_cmc.columns.intersection(pos_col_list)]

    # convert to float32 because pd pivot doesn't work with float16
    for i in pos_col_list:
        if i != "chr_pos":
            Y_trace_pos_noncmc = Y_trace_pos_noncmc.astype({i: np.float32})
            Y_trace_pos_cmc = Y_trace_pos_cmc.astype({i: np.float32})

    # Required Trace and SI values
    Y_pos_list_noncmc, Y_traces_noncmc = categorise_pivot(
        Y_trace_pos_noncmc, "chr_pos", col_list(nn)
    )
    Y_pos_list_cmc, Y_traces_cmc = categorise_pivot(
        Y_trace_pos_cmc, "chr_pos", col_list(nn)
    )

    # plot_trcvals = per_pos_feature('SI_0', Y_sites_dict, Y_pos_list_noncmc, Y_traces_noncmc, Y_pos_list_cmc, Y_traces_cmc)
    plot_trcvals_list = per_pos_feature_list(
        pos_col_list[:-1],
        Y_sites_dict,
        Y_pos_list_noncmc,
        Y_traces_noncmc,
        Y_pos_list_cmc,
        Y_traces_cmc,
    )

    def indices(feat, pos_col_list):
        idx = []
        for i, pos in enumerate(pos_col_list):
            if feat in pos:
                idx.append(i)
        return idx

    def pool_vals(vals_key, indices):
        list_vals = []
        for i in indices:
            list_vals.append(vals_key[i])
        arr = np.array(list_vals)
        return arr

    # Trace value density plot for individual modified positions
    dens = 25
    key = "25s_2924"

    # Calculate KS statistic for each site using their cdf
    def ecdf(sample):

        sample = np.atleast_1d(sample)

        quantiles, counts = np.unique(sample, return_counts=True)
        cumprob = np.cumsum(counts).astype(np.double) / sample.size

        return quantiles, cumprob

    ksv = [[], [], []]  # list recording KS statistic for each site
    std = [[], [], []]  # List to record the stddev of the dists in each key
    skewvals = [[], [], []]
    for key in plot_trcvals_list:
        idx = indices(feat, pos_col_list)
        noncmc_reads = len(idx) * len(plot_trcvals_list[key][0][0])
        cmc_reads = len(idx) * len(plot_trcvals_list[key][1][0])

        # min_reads= min(noncmc_reads,cmc_reads)
        key0_rand = np.array(
            list((pool_vals(plot_trcvals_list[key][0], idx)).flatten())
        )
        key1_rand = np.array(
            list((pool_vals(plot_trcvals_list[key][1], idx)).flatten())
        )

        Y_cdf = ecdf(key0_rand)
        nonY_cdf = ecdf(key1_rand)

        if (cmc_reads / len(idx)) < 20 or (noncmc_reads / len(idx)) < 20:
            continue
        ks, pval = stats.ks_2samp(key0_rand, key1_rand)
        ksv[0].append(key)
        ksv[1].append(ks)
        ksv[2].append(pval)
        std[0].append(np.std(key0_rand))
        std[1].append(np.std(key1_rand))
        std[2].append(key)
        skewvals[0].append(stats.skew(key0_rand))
        skewvals[1].append(stats.skew(key1_rand))
        skewvals[2].append(key)

        Y_trcvals = key0_rand
        nonY_trcvals = key1_rand

        fig = plt.figure(figsize=(6, 9), dpi=100)
        sns.set(font_scale=1.2)
        sns.set_style("white")
        ax = fig.add_subplot(211)
        # ax.hist(Y_trcvals,dens, density=True, alpha = 0.2, color= 'g')
        # ax.hist(nonY_trcvals,dens, density=True, alpha = 0.2, color= 'r')

        bin_heights, bin_borders = np.histogram(Y_trcvals, dens, density=True)
        bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
        ax.fill_between(
            bin_centers,
            bin_heights,
            color="g",
            alpha=0.7,
            label="{}-nonCMC, n={}".format(key, int(len(key0_rand) / len(idx))),
        )

        bin_heights, bin_borders = np.histogram(nonY_trcvals, dens, density=True)
        bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
        ax.fill_between(
            bin_centers,
            bin_heights,
            color="r",
            alpha=0.7,
            label="CMC, n={}".format(int(len(key1_rand) / len(idx))),
        )

        ax.set_xlabel("Signal Intensity")
        ax.set_ylabel("Density")
        # ax.legend(loc='upper center')
        ax.legend(loc="best")

        ax2 = fig.add_subplot(212)

        ax2.plot(Y_cdf[0], Y_cdf[1], color="g", label="{}".format(key), linewidth=2)
        ax2.plot(nonY_cdf[0], nonY_cdf[1], color="r", label="CMC", linewidth=2)
        ax2.plot(
            [], [], label="\nKS = {}\nP={:.1E}".format(round(ks, 2), pval), color="w"
        )

        ax2.set_xlabel("Signal Intensity")
        ax2.set_ylabel("Density")
        # ax.legend(loc='upper center')
        ax2.legend(loc="best")

        fig.tight_layout()

        plt.show()
    return ksv


ksv_tr_Y = ks_extract("TR", Y_pos_mod)
ksv_si_Y = ks_extract("SI", Y_pos_mod)
ksv_tr_HE = ks_extract("TR", HE_pos_mod)
ksv_si_HE = ks_extract("SI", HE_pos_mod)


# Trace and SI KS figure
fig = plt.figure(figsize=(10, 8), dpi=150)
sns.set(font_scale=1)
sns.set_style("white")
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("KS_TR")
ax.set_ylabel("KS_SI")
# ax.set_title('Two component PCA')
ax.scatter(
    ksv_tr_Y[1], ksv_si_Y[1], label="Bonafide \nΨ sites", c="r", marker="o", s=150
)
ax.scatter(
    ksv_tr_HE[1],
    ksv_si_HE[1],
    label="High ME site \n>{}%".format(round(100 * mis_cut, 2)),
    c="b",
    marker="o",
    s=150,
)
ax.plot([], [], label="KS statistic \nvalue", c="w")
ax.legend(loc="best")
plt.show()


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(
        X,
        Y,
        P,
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["dotted", "--", "-"],
    )

    # plot support vectors
    if plot_support:
        ax.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=300,
            linewidth=1,
            facecolors="none",
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


X1 = np.stack((np.array(ksv_tr_Y[1]), np.array(ksv_si_Y[1])), axis=1)
X2 = np.stack((np.array(ksv_tr_HE[1]), np.array(ksv_si_HE[1])), axis=1)
X = np.concatenate((X1, X2), axis=0)
y = np.zeros(len(X))
y[0 : len(X1)] = 1


from sklearn.svm import SVC

C = 10
svm_kernel = "linear"

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    RocCurveDisplay,
)

skf = StratifiedKFold(n_splits=3, shuffle=True)
inds = skf.split(X, y)

avg_scores = []
for i in range(5):
    scores = []
    for train_index, test_index in skf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = SVC(kernel=svm_kernel, C=C, class_weight="balanced").fit(
            X_train, y_train
        )
        score = recall_score(y_test, model.predict(X_test))
        print(round(score, 3))
        scores.append(score)
    avg_kfld_scores = round(np.mean(scores), 3)
    print("Avg = " + str(avg_kfld_scores))
    avg_scores.append(avg_kfld_scores)
print("Average_scores = {}".format(round(np.mean(avg_scores), 3)))


model = SVC(kernel=svm_kernel, C=C, class_weight="balanced").fit(X, y)


# define evaluation procedure
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
# evaluate model
scores = cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
# summarize performance
print("Mean average roc_auc: {}".format(np.mean(scores)))


from skopt.space import Real
from skopt import gp_minimize
from skopt.utils import use_named_args


search_space = []
search_space.append(Real(0.01, 100.0, "log-uniform", name="C"))


@use_named_args(search_space)
def cross_val(C):
    # define evaluation procedure
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import cross_val_score

    svm_model = SVC(kernel=svm_kernel, C=C, class_weight="balanced").fit(X, y)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=None)
    # evaluate model
    scores = cross_val_score(svm_model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    # summarize performance
    print("Mean average score: {}, C = {}".format(np.mean(scores), C))
    return 1 - np.mean(scores)


opt_C = gp_minimize(cross_val, search_space)
# print('Best Recall: %.3f' % (1.0 - opt_C.fun))
print("Best Parameters: {}".format((opt_C.x)))

# model = SVC(kernel=svm_kernel, C=opt_C.x[0], class_weight='balanced').fit(X, y)
model = SVC(kernel=svm_kernel, C=10, class_weight="balanced").fit(X, y)
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)

pred_as_mod = [
    "18s_1009",
    "25s_2725",
    "18s_1186",
    "18s_1710",
    "18s_29",
    "25s_1220",
    "18s_231",
    "18s_1414",
    "25s_1703",
    "25s_2949",
    "18s_710",
    "18s_1286",
    "18s_277",
    "25s_1785",
    "18s_208",
    "18s_101",
    "25s_2056",
    "18s_695",
    "25s_2998",
    "25s_2423",
    "25s_643",
    "25s_987",
    "18s_699",
    "25s_1436",
    "25s_2190",
    "25s_2612",
    "18s_1770",
    "25s_2979",
    "25s_446",
    "25s_2269",
]

coms = list(set(ksv_tr_HE[0]).intersection(pred_as_mod))
mod_indoC_idx = []
for idx, pos in enumerate(ksv_tr_HE[0]):
    if pos in pred_as_mod:
        mod_indoC_idx.append(len(X1) + idx)

cdict = {
    "red": ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)),
    "green": ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.75, 1.0, 1.0), (1.0, 1.0, 1.0)),
    "blue": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)),
}


from copy import deepcopy

y_3_class = deepcopy(y)
for i in mod_indoC_idx:
    y_3_class[i] = 0.75

fig = plt.figure(figsize=(10, 8), dpi=200)
sns.set(font_scale=1)
sns.set_style("white")
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("KS_TR")
ax.set_ylabel("KS_SI")
ax.scatter(X[:, 0], X[:, 1], c=y_3_class, s=120, cmap="coolwarm")
plot_svc_decision_function(model, ax)
ax.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=300,
    lw=1,
    facecolors="none",
)
# ax.set_title('C = {}, best_score = {}'.format(C,round(np.mean(scores),3)), size=14)
# ax.set_title('C = {}, best_score = {}'.format(opt_C.x[0],1-opt_C.fun), size=14)
ax.set_title(
    "Accuracy = {}, ROC-AUC = {}".format(round(acc, 2), round((1 - opt_C.fun), 2)),
    size=12,
)


fig = plt.figure(figsize=(3.7, 3.7), dpi=100)
sns.set(font_scale=1.3)
sns.set_style("white")
ax = fig.add_subplot(1, 1, 1)
fig = RocCurveDisplay.from_estimator(model, X, y, ax=ax)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.legend().set_visible(False)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
plt.show()

print(f"Time: {time.time() - start}")
