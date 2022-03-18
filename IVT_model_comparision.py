#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, matplotlib.pyplot as plt, numpy as np, pandas as pd
import seaborn as sns

sns.set_theme()
sns.set_style("white")
from glob import glob
from common_functions import load_data  # library provided in nanoRMS
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
from trace_extract_funcs import (
    col_list,
    categorise_pivot,
    per_pos_feature,
    per_pos_feature_list,
    get_regions,
)

start = time.time()


# Load data - parameters for nanoRMS functions
nn = 2  # how many neighbour positions to take into account
dt_shift = 10  # expected shift between center of the pore and motor protein in bases
features = ["si", "tr", "dt0", "dt%s" % dt_shift]

fasta = ""  # reference FastA
fnpat_snp = ""  # pattern for all BAM files
fnpat_psu = ""  # pattern for all BAM files


cov_cut = 50
mis_cut = 0.04  # mismatch cutoff

# From csv files generated by Epinano-RMS extract sites for comparision after
# applying coverage and mismatch cutoffs
# Valid U sites - snp -
sites_snp = pd.read_csv("endoc-snp-psu.per.site.baseFreq.csv")
sites_snp["Chr_Position"] = sites_snp["#Ref"] + str("_") + sites_snp["pos"].astype(str)
sites_snp.rename({"#Ref": "Chr", "pos": "Position"}, axis=1, inplace=True)
sites_snp = sites_snp[["Chr_Position", "base", "cov", "mis"]]
chr_pos_sites_snp = sites_snp.query(
    'cov>{} and base=="T" and mis>={}'.format(cov_cut, mis_cut)
)
chr_pos_sites_snp = chr_pos_sites_snp["Chr_Position"].tolist()


# Valid U sites - psu
sites_psu = pd.read_csv("endoc-psu-mod50.per.site.baseFreq.csv")
sites_psu["Chr_Position"] = sites_psu["#Ref"] + str("_") + sites_psu["pos"].astype(str)
sites_psu.rename({"#Ref": "Chr", "pos": "Position"}, axis=1, inplace=True)
sites_psu = sites_psu[["Chr_Position", "base", "cov", "mis"]]
chr_pos_sites_psu = sites_psu.query(
    'cov>{} and base=="T" and mis>={}'.format(cov_cut, mis_cut)
)
chr_pos_sites_psu = chr_pos_sites_psu["Chr_Position"].tolist()

common_sites = list(set(chr_pos_sites_psu).intersection(set(chr_pos_sites_snp)))


sites_dict = {}
for modpos in common_sites:
    sites_dict[modpos] = [modpos]

bams_snp = sorted(glob(fnpat_snp))
bams_psu = sorted(glob(fnpat_psu))


# Utilising trace_df() from nanoRMS
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


mod_regions = get_regions(common_sites)

df_snp = trace_df(fasta, bams_snp, mod_regions, features)
df_psu = trace_df(fasta, bams_psu, mod_regions, features)


# remove all columns except Trace and Signal intensity
pos_col_list = col_list(nn)
pos_col_list.append("chr_pos")
trace_pos_snp = df_snp[df_snp.columns.intersection(pos_col_list)]
trace_pos_psu = df_psu[df_psu.columns.intersection(pos_col_list)]


# convert to float32 because pd pivot doesn't work with float16
for i in pos_col_list:
    if i != "chr_pos":
        trace_pos_snp = trace_pos_snp.astype({i: np.float32})
        trace_pos_psu = trace_pos_psu.astype({i: np.float32})

# Required Trace and SI values
pos_list_snp, traces_snp = categorise_pivot(trace_pos_snp, "chr_pos", col_list(nn))
pos_list_psu, traces_psu = categorise_pivot(trace_pos_psu, "chr_pos", col_list(nn))


plot_trcvals_list = per_pos_feature_list(
    pos_col_list[:-1],
    sites_dict,
    pos_list_snp,
    traces_snp,
    pos_list_psu,
    traces_psu,
)

plot_trcvals = per_pos_feature(
    "TR_0",
    sites_dict,
    pos_list_snp,
    traces_snp,
    pos_list_psu,
    traces_psu,
)


# Trace value density plot for individual modified positions
dens = 25
key = "25s_2735"

# PCA plot of features for each position
for key in plot_trcvals_list:

    x1 = np.stack(plot_trcvals_list[key][0], axis=1)
    x2 = np.stack(plot_trcvals_list[key][1], axis=1)
    x = np.concatenate((x1, x2), axis=0)
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponentsx = pca.fit_transform(x)

    fig = plt.figure(figsize=(10, 8), dpi=70)
    # sns.set(font_scale=0.6)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Two component PCA")
    ax.scatter(
        principalComponentsx.T[0][len(x1) :],
        principalComponentsx.T[1][len(x1) :],
        label="pU Modified",
        c="b",
        marker=".",
        s=2,
        alpha=0.5,
    )
    ax.scatter(
        principalComponentsx.T[0][0 : len(x1)],
        principalComponentsx.T[1][0 : len(x1)],
        label="Unmodified - SNP model",
        c="r",
        marker=".",
        s=3,
        alpha=0.5,
    )
    ax.plot([], [], label="Key - {}".format(key), c="w")
    ax.legend()
    plt.show()


def ecdf(sample):  # Evaluates cdf of input array of values

    sample = np.atleast_1d(sample)

    quantiles, counts = np.unique(sample, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob


key = "endocomporePsuT_64"
ksv = [[], [], []]  # list recording KS statistic for each site
std = [[], [], []]  # List to record the stddev of the dists in each key
skewvals = [[], [], []]
for key in plot_trcvals:
    cdf = ecdf(plot_trcvals[key][0])
    snp_reads = len(plot_trcvals[key][0])
    psu_cdf = ecdf(plot_trcvals[key][1])
    psu_reads = len(plot_trcvals[key][1])
    if psu_reads < 50 or snp_reads < 50:
        continue

    ks, pval = stats.ks_2samp(plot_trcvals[key][0], plot_trcvals[key][1])
    ksv[0].append(key)
    ksv[1].append(ks)
    ksv[2].append(pval)
    std[0].append(np.std(plot_trcvals[key][0]))
    std[1].append(np.std(plot_trcvals[key][1]))
    std[2].append(key)
    skewvals[0].append(stats.skew(plot_trcvals[key][0]))
    skewvals[1].append(stats.skew(plot_trcvals[key][1]))
    skewvals[2].append(key)

    trcvals = plot_trcvals[key][0]
    psu_trcvals = plot_trcvals[key][1]

    fig = plt.figure(figsize=(6, 9), dpi=100)
    sns.set(font_scale=1.2)
    sns.set_style("white")
    ax = fig.add_subplot(211)
    # ax.hist(trcvals,dens, density=True, alpha = 0.2, color= 'g')
    # ax.hist(psu_trcvals,dens, density=True, alpha = 0.2, color= 'r')

    bin_heights, bin_borders = np.histogram(trcvals, dens, density=True)
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    ax.plot([], [], c="w", label="{}".format(key))
    ax.fill_between(
        bin_centers,
        bin_heights,
        color="r",
        alpha=0.7,
        label="Unmodified - SNP model, n={}".format(snp_reads),
    )

    bin_heights, bin_borders = np.histogram(psu_trcvals, dens, density=True)
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    ax.fill_between(
        bin_centers,
        bin_heights,
        color="b",
        alpha=0.7,
        label="pU Modified, n={}".format(psu_reads),
    )

    ax.set_xlabel("Trace Value")
    ax.set_ylabel("Density")
    ax.legend(loc="upper center")
    # ax.legend(loc='best')

    ax2 = fig.add_subplot(212)

    ax2.plot(cdf[0], cdf[1], color="r", label="Unmodified - SNP model", linewidth=2)
    ax2.plot(psu_cdf[0], psu_cdf[1], color="b", label="pU Modified", linewidth=2)
    ax2.plot([], [], label="\nKS = {}\nP={:.1E}".format(round(ks, 2), pval), color="w")

    ax2.set_xlabel("Trace Value")
    ax2.set_ylabel("Density")
    # ax2.legend(loc='upper center')
    ax2.legend(loc="best")

    fig.tight_layout()

    plt.show()


ksv_np = np.array(ksv).T
ksv_pd = pd.DataFrame(ksv_np, columns=["chr_pos", "ksv", "pval"])
ksv_pd["ksv"] = pd.to_numeric(ksv_pd["ksv"], errors="ignore")
ksv_pd = ksv_pd.sort_values("ksv", ascending=(False))

# plot bar plot of KS values in decreasing order
fig = plt.figure(figsize=(8, 2.5), dpi=250)
sns.set(font_scale=0.6)
sns.set_style("white")
ax = fig.add_subplot(111)
ax.bar(ksv_pd["chr_pos"], ksv_pd["ksv"], width=0.7)
ax.set_xlabel("Position")
ax.set_ylabel("KS Statistic value")
ax.tick_params(axis="x", rotation=60, labelsize="small")
fig.tight_layout()
plt.show()

ksv_pd["pval"] = -np.log10((ksv_pd["pval"].astype("float64")))
ksv_pd = ksv_pd.sort_values("pval", ascending=False)

# plot bar plot of KS p-values in decreasing order
fig = plt.figure(figsize=(8, 2.5), dpi=250)
sns.set(font_scale=0.6)
sns.set_style("white")
ax = fig.add_subplot(111)
ax.bar(ksv_pd["chr_pos"], ksv_pd["pval"], width=0.7)
ax.set_xlabel("Position")
ax.set_ylabel("-log p-value for KS statistic")
ax.tick_params(axis="x", rotation=60, labelsize="small")
fig.tight_layout()
plt.show()

# plot distribution of KS values
fig = plt.figure(dpi=250)
ax = fig.add_subplot(111)
ax.set_xlabel("KS value")
ax.set_ylabel("Density")
ax.set_xlim(0, 1)
fig.tight_layout()
sns.set(font_scale=0.8)
sns.set_style("white")
sns.histplot(ksv_pd["ksv"], bins=6, stat="probability", kde=True)
plt.show()


# plot the violin plot for skewness of psu vs snp model
fig = plt.figure(dpi=250)
ax = fig.add_subplot(111)

skewvalsnp = np.array(skewvals)
skewvalsdf = pd.DataFrame(
    {
        "Unmodified - SNP model": skewvalsnp[0],
        "pU Modified": skewvalsnp[1],
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
skewvaldiff = skewvalsnp[1] - skewvalsnp[0]
sns.histplot(skewvaldiff, bins=13, stat="probability", kde=True)
plt.show()

sns.set_style("white")
ax = fig.add_subplot(111)
ax.bar(ksv_pd["chr_pos"], ksv_pd["ksv"], width=0.7)
ax.set_xlabel("Position")
ax.set_ylabel("KS Statistic value")
ax.tick_params(axis="x", rotation=60, labelsize="small")
fig.tight_layout()
plt.show()

ksv_pd["pval"] = -np.log10((ksv_pd["pval"].astype("float64")))
ksv_pd = ksv_pd.sort_values("pval", ascending=False)

# plot bar plot of KS p-values in decreasing order
fig = plt.figure(figsize=(8, 2.5), dpi=250)
sns.set(font_scale=0.6)
sns.set_style("white")
ax = fig.add_subplot(111)
ax.bar(ksv_pd["chr_pos"], ksv_pd["pval"], width=0.7)
ax.set_xlabel("Position")
ax.set_ylabel("-log p-value for KS statistic")
ax.tick_params(axis="x", rotation=60, labelsize="small")
fig.tight_layout()
plt.show()


print(np.mean(ksv[1]))
print(np.std(ksv[1]))
print(f"Time: {time.time() - start}")