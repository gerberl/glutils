"""
10-Aug-24 18:47

Add itertools for pairing elements in a list with different approaches.
Motivated by the need of comparing data distributions in a pairwise fashion
while obtaining centrality and dispersion of comparison metric values.


28-Jun-21 3:11pm

This is the main file for my tiny extension to itertools/functools for my own
projects. At this stage, it has minimal documentation and no testing or
exception handling. At some points, I might address this and make it look much
more like a proper package/module for being used by other people, even if just
for good practice.

Current, it depends on standard `json` module, and Python 3 is assumed so that
things such as dictionary comprehensions work.

---

3-Aug-20: it will be an independent package at some point.

my own helper module for slicing and merging dictionaries, as well as 
doing set operations on tuples
"""

import json
import rich
from functools import partial
from itertools import combinations, product, pairwise


# ## Utility functions for generating pairs of elements with different
#    strategies. Could go on another package (e.g., `disfunctools`)

def pairwise_lst_el_combinations(lst):
    """
    >>> elements = [1, 2, 3, 4, 5]
    >>> list(pairwise_lst_el_combinations(elements))
    [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
    """
    return combinations(lst, 2)

def pairwise_lst_el_chain(lst):
    """
    >>> elements = [1, 2, 3, 4, 5]
    >>> list(pairwise_lst_el_chain(elements))
    [(1, 2), (2, 3), (3, 4), (4, 5)]
    """
    return pairwise(lst)

def pairwise_lst_1st_vs_rest(lst):
    """
    >>> elements = [1, 2, 3, 4, 5]
    >>> list(pairwise_lst_1st_vs_rest(elements))
    [(1, 2), (1, 3), (1, 4), (1, 5)]
    """
    first = lst[0:1]
    rest = lst[1:]
    return product(first, rest)




def slice_dict(my_dict, keys): 
    if not isinstance(keys, (list, tuple)):
        # works for a few cases - a single string is wrapped into a list
        keys = [keys] 
    return { k: v for k, v in my_dict.items() if k in keys }


def dict_drop(my_dict, keys):
    """
    You've guessed it right - returns a new dictionary with `keys` 
    removed from `my_dict`
    """
    if not isinstance(keys, (list, tuple)):
        # works for a few cases - a single string is wrapped into a list
        keys = [keys] 
    return { k: v for k, v in my_dict.items() if k not in keys }


def merge_dicts(d1, d2):
    return { **d1, **d2 }


def tuple_minus(t1, t2):
    return tuple(set(t1) - set(t2))


def key_tuple(t, k):
    # k is a single value, so it has to be cast into a tuple with the (,)
    # construct
    return tuple_minus(t, (k,))


# these only work if `k` is a single value; it will break for compound keys
def key_dict(d, k):
    """>>> key_dict({ 'a': 1, 'b': 2, 'c': 3}, 'a')
    (1, {'b': 2, 'c': 3})"""
    return (
        d[k], 
        slice_dict(
            d, 
            tuple_minus(d.keys(), (k,))
        )
    )


def pprint_dict(my_dict, use_rich=True):
    print_function = rich.print if use_rich else print

    print(json.dumps(my_dict, indent=4))