#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random


def col_list(nn):
    """
    Get names for the columns containing Trace value and Signal intensity value
    for a given nearest number of bases, from the dataframe returned by trace_df

    Parameters
    ----------
    nn : int for number of nearest neighbours

    Returns
    -------
    list : list containing column names
    """
    list = []
    for i in range(2 * nn + 1):
        list.append("SI_" + str(i - nn))
    for i in range(2 * nn + 1):
        list.append("TR_" + str(i - nn))
    return list


def categorise_pivot(df, val, item):
    """
    Generates pivot of df with val as axis (in this case usually chr_pos)

    Parameters
    ----------
    df : dataframe
        dataframe like the one returned by trace_df

    val : string
        column name for the axis of the pivot

    item : list
        list of column names to be pivoted (in this case like the one returned by col_list())

    Returns
    -------
    list
        list containing a list of the item column names and an array of val category values

    out : list
        list of lists (in the order dictated by the list item), with each
        element list being a list of arrays of pivoted values (in this case
        Trace values and/or SI values)
    """

    item.sort()
    # print(item)
    out = []
    for pos in item:
        df_unique = pd.unique(df[str(val)])
        piv = df.pivot(columns=str(val), values=str(pos))
        arr_list = np.transpose(piv.to_numpy())
        arr_list = [arr for arr in arr_list]
        out.append(
            [arr[~np.isnan(arr)] for arr in arr_list]
        )  # remove NaNs introduced by pivot

    return [item, df_unique], out


from Bio import SeqIO


def seq_dict(fastapath):
    """
    Creates dictionary of fasta entries, with id as key and sequence as value

    Parameters
    ----------
    fastapath : path to the fasta file

    Returns
    -------
    chrseq : dictionary
    """
    records = list(SeqIO.parse(fastapath, "fasta"))
    chrseq = {}
    for i in records:
        chrseq[i.id] = str(i._seq)
    return chrseq


def get_nn(seqdict, positions, nn, base):
    """
    Returns nn nearest neigbour bases on either side of the imput positions (i.e. a 2nn+1 mer)

    Parameters
    ----------
    seqdict : dictionary in the format returned by seq_dict()

    positions : list containing positions in the format id_position or chr_position

    nn : int - number of nearest neighbours

    base : either a string (enter single base) or None
            Replaces the middle position of the kmer with the base indicated
            in base and adds it in addition to identical kmer

    Returns
    -------
    nmers : list of nmer strings
    """

    nmers = []
    for pos in positions:
        loc = pos.split("_")
        from copy import deepcopy

        join = deepcopy(loc)
        del join[-1]
        join = "_".join(join)
        seq = seqdict[join]
        if ((int(loc[-1]) - (nn + 1)) < 0) or ((int(loc[-1]) + nn) > len(seq)):
            print("skipped site {}".format(loc[-1]))  # edge of sequence no neighbours
            continue
        nmer = seq[(int(loc[-1]) - (nn + 1)) : (int(loc[-1]) + nn)]
        nmers.append(nmer)
        if base != None:
            nmer_baseswitch = list(nmer)
            nmer_baseswitch[nn] = str(base)
            nmer_baseswitch = "".join(nmer_baseswitch)
            nmers.append(nmer_baseswitch)

    return nmers


def get_nn_pos(seqdict, positions, nn, base):
    """
    Returns nn nearest neigbour bases on either side of the input positions (i.e. a 2nn+1 mer)

    Parameters
    ----------
    seqdict : dictionary in the format returned by seq_dict()

    positions : list containing positions in the format id_position or chr_position

    nn : int - number of nearest neighbours

    base : either a string (enter single base) or None
            Replaces the middle position of the kmer with the base indicated
            in base and adds it in addition to identical kmer
    Returns
    -------
    nmers : list of nmer strings, and associated positions
    """
    poslist = []
    nmers = []
    for pos in positions:
        loc = pos.split("_")
        from copy import deepcopy

        join = deepcopy(loc)
        del join[-1]
        join = "_".join(join)
        seq = seqdict[join]
        if ((int(loc[-1]) - (nn + 1)) < 0) or ((int(loc[-1]) + nn) > len(seq)):
            print("skipped site {}".format(loc[-1]))
            continue
        nmer = seq[(int(loc[-1]) - (nn + 1)) : (int(loc[-1]) + nn)]
        nmers.append(nmer)
        if base != None:
            nmer_baseswitch = list(nmer)
            nmer_baseswitch[nn] = str(base)
            nmer_baseswitch = "".join(nmer_baseswitch)
            nmers.append(nmer_baseswitch)
        poslist.append(pos)

    return poslist, nmers


def get_regions(chr_pos):
    """
    Modified get_regions() to work with reference names that include an underscore
    Does the same thing as the usual get_regions from nanoRMS except, preserves the
    actual reference name, instead of only the first element after _ split

    Parameters
    ----------
    chr_pos : list
        List of character positions in the format "Reference-name_position"

    Returns
    -------
    regions : list of tuples
        Each tuple corresponds to its position in the format required by trace_df()

    """

    regions = [
        ("_".join(cp.split("_")[:-1]), int(cp.split("_")[-1]), "_") for cp in chr_pos
    ]
    return regions


def findall(p, s):
    """Yields all the positions of
    the pattern p in the string s in 0-based indexing"""
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i + 1)


def nn_pos(pos, nn=2):
    """
    Returns list of nn neighbouring positions of 'pos'

    Parameters
    ----------
    pos : str
        Input position in the format Chr_Position

    nn : int, optional
        Number of neighbouring positions. The default is 2.

    Returns
    -------
    nn_list : list
        list of nn neighbouring positions with input pos in the centre

    """

    regions = get_regions([pos])[0]
    start = regions[1] - nn
    nn_list = []
    for i in range(2 * nn + 1):
        nn_list.append(str(regions[0] + str("_") + str(start + i)))

    return nn_list


def purge_mod(pos_list, ban_list, nn=2):
    """
    If any of the nn neighbouring positions for a given pos in pos_list
    appear in ban_list, then pos is removed from pos_list

    Parameters
    ----------
    pos_list : list
        list of positions each in the format Chr_Position

    ban_list : list
        list of disallowed positions

    nn : int, optional
        number of neighbouring positions to check. The default is 2.

    Returns
    -------
    list
        pos_list updated with the offending positions removed

    """

    purge_list = []
    for pos in pos_list:
        if bool(set(nn_pos(pos, nn)).intersection(ban_list)):
            purge_list.append(pos)

    return list(set(pos_list).difference(purge_list))


def search_Pos(seqdict, lookup, ban_pos=None):
    """
    search for sequence input in the dictionary of reference sequences

    Parameters
    ----------
    seqdict : dictionary in the format returned by seq_dict()

    lookup : string for the sequence for eg. in the format returned by get_nn(),
        should have odd length

    ban_pos : list, optional
        list of disallowed positions for use in internal purge_mod()

    Returns
    -------
    pos : list of centre positions of sequence matches
    """
    if (len(lookup) % 2) == 0:
        raise ValueError("n-mer should have odd number of bases!")

    nn = (len(lookup) - 1) / 2
    pos = []
    for key, value in seqdict.items():
        if lookup in value:
            match_pos = [
                str(key) + "_" + str(int(k + nn + 1)) for k in findall(lookup, value)
            ]
            if ban_pos != None:
                match_pos = purge_mod(match_pos, ban_list=ban_pos, nn=int(nn))
            pos = pos + match_pos
    return pos


def search_multiPos(seqdict, lookup, ban_pos=None):
    """
    same as search_Pos but takes a list of sequence queries as input

    Parameters
    ----------
    seqdict : dict
        dictionary in the format returned by seq_dict()

    lookup : str
        list of strings for use in internal search_Pos()

    ban_pos : list, optional
        list of disallowed positions for use in internal search_Pos(). The default is None.

    Returns
    -------
    pos : list
        list of centre positions of sequence matches

    """
    pos = []
    for nmer in lookup:
        pos = pos + search_Pos(seqdict, nmer, ban_pos=ban_pos)
    return pos


def search_multiPos_link(seqdict, positions, nn, base, source_seq=None, ban_pos=None):
    """
    similar to search_multiPos
    finds positions with the same kmer sequence as input positions

    Parameters
    ----------
    seqdict : dict
        dictionary in the format returned by seq_dict()

    positions : list
        list containing positions in the format id_position or chr_position

    nn : int
        number of nearest neighbours

    base : either a string (enter single base) or None
        see entry for get_nn_pos()

    source_seq : dict, optional
        sequence within which the kmer sequences are extracted from (if None, same as seqdict )
        The default is None.

    ban_pos : list, optional
        list of disallowed positions for use in internal search_Pos(). The default is None.

    Returns
    -------
    posdict : dict
        dictionary with an individual input position as key, and list of
        positions in seqdict, with kmer sequence match as value

    """

    if source_seq == None:
        source_seq = seqdict

    pos, nmers = get_nn_pos(
        source_seq, positions, nn, base
    )  # to ensure pos, nmers are always same length

    lookup = list(zip(pos, nmers))
    posdict = {}
    for pos, nmer in lookup:
        posdict[pos] = search_Pos(seqdict, nmer, ban_pos=ban_pos)
    return posdict


def dict_filter(posdict, filterset):
    """
    Applies a filter on the values of the input dictionary
    Values in the dictionary are lists whose elements that aren't present in
    the filterset are removed

    Parameters
    ----------
    posdict : dictionary of the format returned by search_multiPos_link()

    filterset : list containing positions in the format id_position or chr_position

    Returns
    -------
    return_dict : dictionary in the same format as postdict and preserved keys,
    with lists in the values filtered
    """
    return_dict = {}
    for key, value in posdict.items():
        filterlist = list((set(filterset).intersection(set(value))))
        for k in posdict:
            if k in filterlist:
                filterlist.remove(str(k))
        return_dict[key] = filterlist
    return return_dict


def mean_feat(col_list, feature, feat_table):
    """
    trace_df() can output for eg. TR and SI values off nn nearest neighbour
    positions. This function calculates the mean of all neighbouring positions
    and also the zero position and returns a pandas Series of the means for
    each read

    Parameters
    ----------
    col_list : list
        List of strings, each of which is the name of the feature columns
        present in the table outputted by trace_df() in get_features.py
        for eg. SI_0, TR_-1 etc

    feature : str
        string which is an element of col_list without the position number, of
        which the mean is going to be calculated eg SI in case of calculation
        of mean of signal intensity for all positions

    feat_table : pandas DataFrame
        Dataframe of the kind returned by trace_df()

    Returns
    -------
    mean : pandas Series
        Series which is a column of the mean values of the required feature
        across all positions (-nn,nn)

    """
    cols = []
    for feat in col_list:
        if feature in feat:
            cols.append(feat)
    feat_table = feat_table[feat_table.columns.intersection(cols)]
    mean = feat_table.mean(axis=1)
    mean = mean.rename(feature + "_mean")
    return mean


def mean_SITR_concat(pos_col_list, trace_pos):
    """
     Calculates just SI and TR mean columns for all neighbouring positions and
     concatenates them to give a dataframe of mean columns with reads as rows

     Parameters
     ----------
    pos_col_list : list
         List of strings, each of which is the name of the feature columns
         present in the table outputted by trace_df() in get_features.py
         for eg. SI_0, TR_-1 etc

     trace_pos : pandas DataFrame
         Dataframe of the kind returned by trace_df()

     Returns
     -------
     mean_feat_df : pandas DataFrame
         Dataframe with concatenated SI and TR mean columns, appended with
         associated chr_pos

    """
    SI_mean = mean_feat(pos_col_list, "SI", trace_pos)
    TR_mean = mean_feat(pos_col_list, "TR", trace_pos)
    mean_pos_list = trace_pos["chr_pos"]
    mean_feat_df = pd.concat([SI_mean, TR_mean, mean_pos_list], axis=1)
    return mean_feat_df


def mean_feat_perpos(mean_feat_df, sites_list):
    """
    Calculates the mean of the values in the column across all reads of every
    chr_pos

    Parameters
    ----------
    mean_feat_df : pandas DataFrame
        dataframe of the kind output by mean_SITR_concat

    sites_list : list
        list of sites for which the mean across reads needs to be calculated

    Returns
    -------
    mean_df : dataframe
        Dataframe with means across reads for every position in sites_list
        THe chr_pos column is removed and the order of means is the same as
        the order of positions in sites_list
        NOTE : Values get converted to strings for some reason

    """
    site_means = []
    for site in sites_list:
        print(site)
        site_df = mean_feat_df.query("chr_pos == '{}'".format(site))
        site_df = site_df.drop("chr_pos", axis=1)
        site_means.append(site_df.mean())
    mean_df = pd.concat(site_means, axis=1)
    mean_df = mean_df.transpose()
    # mean_df = mean_df.apply(pd.to_numeric)
    return mean_df


def per_pos_feature(
    feature, kmer_sites, mod_pos_list, mod_traces, kmer_pos_list, kmer_traces
):
    """
    Get systematically formatted data structure of feature values (SI, TR etc.)
    of primary sites and associated comparision sites (values are aggregated)

    Parameters
    ----------
    feature : str
        The feature for which values are to be extracted

    kmer_sites : dict
        Dictionary of the kind returned by search_multiPos_link()

    mod_pos_list : list
        list of the kind returned by categorise_pivot() (first returned list)
        This is for the primary sites

    mod_traces : list of lists of numpy arrays
        of the kind returned by categorise_pivot() (out)
        This is for the primary sites

    kmer_pos_list : list
        list of the kind returned by categorise_pivot() (first returned list)
        This is for the comparision sites

    kmer_traces : list of lists of numpy arrays
            of the kind returned by categorise_pivot() (out)
            This is for the comparision sites

    Returns
    -------
    master_dict : dict
        Each key is the primary site name
        Contains two numpy arrays as values
        First array is a 1d array of feature values of primary sites
        Second one is a 1d array of feature values of its associated comparision sites
        (generated by concatenation of 1d arrays of feature values for each
         associated comparision site)

    """
    master_dict = {}

    for modpos, kmerpos in kmer_sites.items():
        if not kmerpos:  # modpos with no identical kmer sites in reference
            continue
        modposindex = np.where(mod_pos_list[1] == modpos)[0][0]
        traceindex = mod_pos_list[0].index(feature)

        kmertrc = []
        for pos in kmerpos:
            if len(np.where(kmer_pos_list[1] == pos)[0]) == 0:
                continue  # On the off chance that some random kmer is left over
            posindex = np.where(kmer_pos_list[1] == pos)[0][0]

            kmertrc.append(kmer_traces[traceindex][posindex])
        kmertrc_arr = np.concatenate(kmertrc)
        modtrc = mod_traces[traceindex][modposindex]
        # if no reads then remove those positions from both sets
        if len(modtrc) != 0 and len(kmertrc_arr) != 0:
            master_dict[modpos] = [modtrc, kmertrc_arr]

    return master_dict


def per_pos_feature_list(
    features, kmer_sites, mod_pos_list, mod_traces, kmer_pos_list, kmer_traces
):
    """
    Like per_pos_feature(), but 'features' is now a list of features
    instead of the single feature input that the one above takes

    Returns
    -------
    master_dict : dict
        Similar to the on returned by per_pos_feature(), but instead of each key
        having 2 numpy arrays as values, it now has 2 lists, corresponding to the primary
        and comparision sites, the elements of which are 1d arrays of the feature values
        appearing in the order they do in 'features'

    """
    master_dict = {}
    for modpos, kmerpos in kmer_sites.items():
        if not kmerpos:  # modpos with no identical kmer sites in reference
            continue
        modposindex = np.where(mod_pos_list[1] == modpos)[0][0]
        featindices = [mod_pos_list[0].index(feat) for feat in features]

        kmerfeattrc = []
        for featindex in featindices:
            kmertrc = []
            for pos in kmerpos:
                if len(np.where(kmer_pos_list[1] == pos)[0]) == 0:
                    print("Skipped - " + str(pos) + " " + str(len(kmertrc)))
                    continue  # On the off chance that some random kmer is left over
                posindex = np.where(kmer_pos_list[1] == pos)[0][0]
                kmertrc.append(kmer_traces[featindex][posindex])
            if len(kmertrc) != 0:
                kmertrc_arr = np.concatenate(kmertrc)
                kmerfeattrc.append(kmertrc_arr)
        modfeattrc = [mod_traces[featindex][modposindex] for featindex in featindices]

        # if no reads then remove those positions from both sets
        if len(modfeattrc) != 0 and len(kmerfeattrc) != 0:
            master_dict[modpos] = [modfeattrc, kmerfeattrc]

    return master_dict


def compare_sites(
    Y_pos_mod,
    ref,
    sites_table,
    nn,
    base,
    cov_cut,
    all_pos_mod,
    mis_cut=None,
    samp=None,
    ban_pos=None,
):
    sites = sites_table.copy()
    sites["Chr_Position"] = sites["#Ref"] + str("_") + sites["pos"].astype(str)
    sites.rename({"#Ref": "Chr", "pos": "Position"}, axis=1, inplace=True)
    sites = sites[["Chr_Position", "base", "cov", "mis"]]
    if mis_cut == None:
        sites = sites.query(
            'cov>{} and (base=="T" or base=="C")'.format(cov_cut)
        )  # All U sites
    else:
        sites = sites.query(
            'cov>{} and (base=="T"or base=="C") and mis<={}'.format(cov_cut, mis_cut)
        )  # All U sites

    # All positions with same kmer sequence as Ymod positions
    kmers = get_nn(ref, Y_pos_mod, nn, base=base)
    # print(kmers)
    kmer_pos_pool = search_multiPos(ref, kmers, ban_pos=ban_pos)
    kmer_pos = search_multiPos_link(ref, Y_pos_mod, nn, base=base, ban_pos=ban_pos)

    chr_pos_sites = sites["Chr_Position"].tolist()
    # Only take those U sites which have the same k mer as one of the bonafide Ymod site
    kmer_sites_pool = list(
        (set(chr_pos_sites).intersection(set(kmer_pos_pool)))
        .difference(set(all_pos_mod))
        .difference(set(Y_pos_mod))
    )
    if samp != None:
        kmer_sites_pool = random.sample(
            kmer_sites_pool, int(len(kmer_sites_pool) / samp)
        )
    kmer_sites = dict_filter(kmer_pos, kmer_sites_pool)
    # flattens the dictionary values
    kmer_sites_values = [
        item for sublist in list(kmer_sites.values()) for item in sublist
    ]

    return kmer_sites, kmer_sites_values
