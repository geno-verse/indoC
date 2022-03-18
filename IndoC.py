#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, matplotlib.pyplot as plt, numpy as np, pandas as pd
import seaborn as sns

sns.set_theme()
sns.set_style("white")
from glob import glob
from common_functions import load_data  # from nanoRMS
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
from trace_extract_funcs import (
    seq_dict,
    col_list,
    categorise_pivot,
    per_pos_feature,
    per_pos_feature_list,
    compare_sites,
)

start = time.time()


# Load data
nn = 2  # how many neighbour positions to take into account
dt_shift = 10  # expected shift between center of the pore and motor protein in bases
features = ["si", "tr", "dt0", "dt%s" % dt_shift]
base = "C"

fasta = "yeast_rRNA.fa"  # reference FastA
ref = seq_dict(fasta)
fnpat = "*.bam"  # pattern for all BAM files


modpos = pd.read_csv(
    "RNA_Mod_Positions_rRNAYeast.tsv",
    sep="\t",
)
all_pos_mod = modpos.query('Mod!="Unm"')
all_pos_mod = all_pos_mod["Chr_Position"].to_list()

cov_cut = 200
alt_mis_cut = 0.05

# positions of all Y modifications
Ymodpos = modpos.query('Mod=="Y"')
Ymodpos = Ymodpos[["Chr_Position"]]
Y_pos_mod = Ymodpos["Chr_Position"].tolist()


# positions of all random ME pos
sites = pd.read_csv("yeast_rRNA.per.site.baseFreq.csv")
sites["Chr_Position"] = sites["#Ref"] + str("_") + sites["pos"].astype(str)
sites.rename({"#Ref": "Chr", "pos": "Position"}, axis=1, inplace=True)
sites = sites[["Chr_Position", "base", "cov", "mis"]]
sites = sites.query(
    'cov>{} and base=="T" and mis>={}'.format(cov_cut, alt_mis_cut)
)  # All U sites
sites = sites["Chr_Position"].tolist()
sites = list(set(sites).difference(all_pos_mod))

rand_sites = sites


# Valid U sites
sites = pd.read_csv("yeast_rRNA.per.site.baseFreq.csv")

kmer_sites_Y, kmer_sites_pool_Y = compare_sites(
    Y_pos_mod,
    ref,
    sites,
    nn,
    base,
    cov_cut,
    all_pos_mod,
    mis_cut=alt_mis_cut,
    ban_pos=all_pos_mod,
)
kmer_sites_rand, kmer_sites_pool_rand = compare_sites(
    rand_sites,
    ref,
    sites,
    nn,
    base,
    cov_cut,
    all_pos_mod,
    mis_cut=alt_mis_cut,
    ban_pos=all_pos_mod,
)


bams = sorted(glob(fnpat))


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


def get_params(Y_pos_mod, kmer_sites, kmer_sites_pool, fasta, bams, features, nn):

    mod_regions = get_regions(Y_pos_mod)

    mod_regions = get_regions(Y_pos_mod)
    kmer_regions = get_regions(kmer_sites_pool)

    # df with TR and SI of all Y mods
    df_Y = trace_df(fasta, bams, mod_regions, features)
    # df with TR and SI of all positions with kmer sequence same as Ymodpos
    df_kmer = trace_df(fasta, bams, kmer_regions, features)

    # remove all columns except Trace and Signal intensity
    pos_col_list = col_list(nn)
    pos_col_list.append("chr_pos")
    Y_trace_pos = df_Y[df_Y.columns.intersection(pos_col_list)]
    kmer_trace_pos = df_kmer[df_kmer.columns.intersection(pos_col_list)]

    # convert to float32 because pd pivot doesn't work with float16
    for i in pos_col_list:
        if i != "chr_pos":
            Y_trace_pos = Y_trace_pos.astype({i: np.float32})
            kmer_trace_pos = kmer_trace_pos.astype({i: np.float32})

    # Required Trace and SI values
    Y_pos_list, Y_traces = categorise_pivot(Y_trace_pos, "chr_pos", col_list(nn))
    kmer_pos_list, kmer_traces = categorise_pivot(
        kmer_trace_pos, "chr_pos", col_list(nn)
    )

    plot_trcvals_list = per_pos_feature_list(
        pos_col_list[:-1], kmer_sites, Y_pos_list, Y_traces, kmer_pos_list, kmer_traces
    )

    plot_trcvals = per_pos_feature(
        "TR_0", kmer_sites, Y_pos_list, Y_traces, kmer_pos_list, kmer_traces
    )

    # Trace value density plot for individual modified positions
    dens = 25
    key = "5s_50"

    # Calculate KS statistic for each site using their cdf
    def ecdf(sample):

        sample = np.atleast_1d(sample)

        quantiles, counts = np.unique(sample, return_counts=True)
        cumprob = np.cumsum(counts).astype(np.double) / sample.size

        return quantiles, cumprob

    ksv = [[], [], []]  # list recording KS statistic for each site
    std = [[], [], []]  # List to record the stddev of the dists in each key
    skewvals = [[], [], []]
    for key in plot_trcvals:
        Y_cdf = ecdf(plot_trcvals[key][0])
        Y_reads = len(plot_trcvals[key][0])
        nonY_cdf = ecdf(plot_trcvals[key][1])
        nonY_reads = len(plot_trcvals[key][1])
        if Y_reads < cov_cut or nonY_reads < cov_cut:
            continue

        def mean_dict(vals_key, indices):
            list_vals = []
            for i in indices:
                list_vals.append(vals_key[i])
            arr = np.array(list_vals)
            return np.mean(arr, axis=0)

        ks_si, pval = stats.ks_2samp(
            plot_trcvals_list[key][0][nn], plot_trcvals_list[key][1][nn]
        )
        ks_tr, pval = stats.ks_2samp(
            plot_trcvals_list[key][0][3 * nn + 1], plot_trcvals_list[key][1][3 * nn + 1]
        )
        # indices_si = [2]
        # indices_tr = [7]
        # ks_si,pval = stats.ks_2samp(mean_dict(plot_trcvals_list[key][0],indices_si),\
        #                             mean_dict(plot_trcvals_list[key][1],indices_si))
        # ks_tr,pval = stats.ks_2samp(mean_dict(plot_trcvals_list[key][0],indices_tr),\
        #                             mean_dict(plot_trcvals_list[key][1],indices_tr))
        ksv[0].append(key)
        ksv[1].append(ks_tr)
        ksv[2].append(ks_si)
        std[0].append(np.std(plot_trcvals[key][0]))
        std[1].append(np.std(plot_trcvals[key][1]))
        std[2].append(key)
        skewvals[0].append(stats.skew(plot_trcvals[key][0]))
        skewvals[1].append(stats.skew(plot_trcvals[key][1]))
        skewvals[2].append(key)

        Y_trcvals = plot_trcvals[key][0]
        nonY_trcvals = plot_trcvals[key][1]

    ksv_np = np.array(ksv).T
    ksv_pd = pd.DataFrame(ksv_np, columns=["chr_pos", "ks_tr", "ks_si"])
    ksv_pd["ks_tr"] = pd.to_numeric(ksv_pd["ks_tr"], errors="ignore")
    ksv_pd["ks_si"] = pd.to_numeric(ksv_pd["ks_si"], errors="ignore")
    ksv_pd = ksv_pd.sort_values("ks_tr", ascending=(False))

    # plot bar plot of KS values in decreasing order
    fig = plt.figure(figsize=(8, 2.5), dpi=250)
    sns.set(font_scale=0.6)
    sns.set_style("white")
    ax = fig.add_subplot(111)
    ax.bar(ksv_pd["chr_pos"], ksv_pd["ks_tr"], width=0.7)
    ax.set_xlabel("Position")
    ax.set_ylabel("KS Statistic value")
    ax.tick_params(axis="x", rotation=60, labelsize="small")
    fig.tight_layout()
    plt.show()
    ksv_pd = ksv_pd.sort_index()

    # plot distribution of KS values
    fig = plt.figure(dpi=250)
    ax = fig.add_subplot(111)
    ax.set_xlabel("KS value")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    sns.set(font_scale=0.8)
    sns.set_style("white")
    sns.histplot(ksv_pd["ks_tr"], bins=6, stat="probability", kde=True)
    plt.show()

    # plot the violin plot for skewness
    fig = plt.figure(dpi=250)
    ax = fig.add_subplot(111)

    skewvalsnp = np.array(skewvals)
    skewvalsdf = pd.DataFrame(
        {
            "Identical5-mers >5% ME - nomods": skewvalsnp[1],
            " random T sites >5%ME": skewvalsnp[0],
            "chr_pos": skewvalsnp[2],
        }
    )
    skewvalsdf = skewvalsdf.melt(
        "chr_pos", var_name="Modification Status", value_name="Skewness"
    )
    for col in skewvalsdf:
        skewvalsdf[col] = pd.to_numeric(skewvalsdf[col], errors="ignore")

    pal = sns.color_palette("hls", len(skewvalsnp.T))
    sns.swarmplot(
        x="Modification Status",
        y="Skewness",
        hue="chr_pos",
        palette=pal,
        size=6,
        data=skewvalsdf,
    )
    sns.violinplot(
        x="Modification Status",
        y="Skewness",
        color="whitesmoke",
        data=skewvalsdf,
        inner=None,
    )
    ax.set_xlabel("")
    ax.legend([])
    plt.show()

    # Plot distribution of difference in skewness
    fig = plt.figure(dpi=250)
    ax = fig.add_subplot(111)
    ax.set_xlabel("Skewness Difference")
    skewvalsnp = (np.delete(skewvalsnp, 2, 0)).astype("float64")
    skewvaldiff = skewvalsnp[0] - skewvalsnp[1]
    sns.histplot(skewvaldiff, bins=13, stat="probability", kde=True)
    plt.show()

    return [ksv_pd, skewvaldiff]


params_bony = get_params(
    Y_pos_mod, kmer_sites_Y, kmer_sites_pool_Y, fasta, bams, features, nn
)
params_rand = get_params(
    rand_sites, kmer_sites_rand, kmer_sites_pool_rand, fasta, bams, features, nn
)


# PCA plot of features for each position

bonY_list = [
    params_bony[0]["ks_tr"].tolist(),
    params_bony[0]["ks_si"].tolist(),
    params_bony[1].tolist(),
]
rand_list = [
    params_rand[0]["ks_tr"].tolist(),
    params_rand[0]["ks_si"].tolist(),
    params_rand[1].tolist(),
]

x1 = np.stack(bonY_list, axis=1)
x2 = np.stack(rand_list, axis=1)
x = np.concatenate((x1, x2), axis=0)
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponentsx = pca.fit_transform(x)

X_chr_pos = np.concatenate(
    (np.array(params_bony[0]["chr_pos"]), np.array(params_rand[0]["chr_pos"])), axis=0
)


fig = plt.figure(figsize=(10, 6), dpi=200)
sns.set(font_scale=0.6)
sns.set_style("white")
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
# ax.set_title('Two component PCA')
# targets = ['Unmodified', 'Modified']
ax.scatter(
    principalComponentsx.T[0][len(x1) :],
    principalComponentsx.T[1][len(x1) :],
    label="High ME >{}%".format(int(100 * alt_mis_cut)),
    c="b",
    marker="o",
    s=50,
)
ax.scatter(
    principalComponentsx.T[0][0 : len(x1)],
    principalComponentsx.T[1][0 : len(x1)],
    label="Bonafide Ψ sites",
    c="r",
    marker="o",
    s=50,
)
ax.plot(
    [],
    [],
    label="PCA of KS (TR and SI) \n& skewness vs \nidentical 5-mers",
    c="w",
)
ax.legend(loc="best")
# ax.legend(loc="lower right")

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


X = principalComponentsx
y = np.zeros(len(X))
y[0 : len(x1)] = 1


cmc_min_sites = pd.read_csv("yeast_minusCMC.per.site.baseFreq.csv")
cmc_min_sites["#Ref"] = cmc_min_sites["#Ref"].str.replace("_yeast", "")
cmc_min_sites["Chr_Position"] = (
    cmc_min_sites["#Ref"] + str("_") + cmc_min_sites["pos"].astype(str)
)
cmc_min_sites.rename({"#Ref": "Chr", "pos": "Position"}, axis=1, inplace=True)
cmc_min_sites = cmc_min_sites[["Chr_Position", "base", "cov", "mis"]]
cmc_min_sites = cmc_min_sites.query(
    'cov>{} and base=="T" and mis>={}'.format(cov_cut, alt_mis_cut)
)
cmc_min_sites = cmc_min_sites["Chr_Position"].tolist()

import copy

y_3class = copy.deepcopy(y)
true_snp = list(set(params_rand[0]["chr_pos"].tolist()).intersection(cmc_min_sites))
rand_list_sites = params_rand[0]["chr_pos"].tolist()
true_snp_idx = []
for idx, site in enumerate(rand_list_sites):
    if site in true_snp:
        true_snp_idx.append(idx)
for i in true_snp_idx:
    y_3class[len(x1) + i] = 0.2


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
scores = cross_val_score(model, X, y, scoring="recall", cv=cv, n_jobs=-1)
# summarize performance
print("Mean average recall: {}".format(np.mean(scores)))


from skopt.space import Real
from skopt import gp_minimize
from skopt.utils import use_named_args


search_space = []
search_space.append(Real(5e-2, 10e2, "log-uniform", name="C"))


@use_named_args(search_space)
def cross_val(C):
    # define evaluation procedure
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import cross_val_score

    svm_model = SVC(kernel=svm_kernel, C=C, class_weight="balanced").fit(X, y)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    # evaluate model
    scores = cross_val_score(svm_model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    # summarize performance
    print("Mean average roc_auc: {}, C = {}".format(np.mean(scores), C))
    return 1 - np.mean(scores)


opt_C = gp_minimize(cross_val, search_space)
print("Best ROC-AUC: %.3f" % (1.0 - opt_C.fun))
print("Best Parameters: {}".format((opt_C.x)))

model = SVC(kernel=svm_kernel, C=opt_C.x[0], class_weight="balanced").fit(X, y)

y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)

fig = plt.figure(figsize=(10, 6), dpi=200)
sns.set(font_scale=0.6)
sns.set_style("white")
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.scatter(X[:, 0], X[:, 1], c=y, s=90, cmap="coolwarm")
plot_svc_decision_function(model, ax)
# ax.scatter(model.support_vectors_[:, 0],
#         model.support_vectors_[:, 1],
#         s=50, lw=1, facecolors='g');
# ax.set_title('C = {}, best_score = {}'.format(C,round(np.mean(scores),3)), size=14)
ax.set_title(
    "Accuracy = {}, ROC-AUC = {}".format(round(acc, 2), round((1 - opt_C.fun), 2)),
    size=10,
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


pred_as_mod = list(X_chr_pos[np.array(y_pred, dtype="bool")])
pred_as_mod = list(set(pred_as_mod).difference(list(params_bony[0]["chr_pos"])))


print(f"Time: {time.time() - start}")
