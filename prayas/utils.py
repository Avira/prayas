import numpy as np
import datetime

from copy import deepcopy


def comb2(s):
    return np.transpose([np.tile(s, len(s)), np.repeat(s, len(s))])


def comb_mat(s):
    rotate = np.arange(0, len(s))
    return np.asarray([list(np.roll(s, -i)) for i in rotate])


def today():
    return datetime.date.today()


def flatten_list(nested_list):
    """
    Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]

    Copied from https://gist.github.com/Wilfred/7889868
    """
    nested_list = deepcopy(nested_list)

    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist

def rle(inarray):
    """
    Run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)

    Copied from https://stackoverflow.com/a/32681075
    """
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, ia[i])
