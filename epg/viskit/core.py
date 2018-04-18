import itertools
import json
import os

import numpy as np


# from sandbox.rocky.utils.py_utils import AttrDict

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def unique(l):
    return list(set(l))


def flatten(l):
    return [item for sublist in l for item in sublist]


def load_progress(progress_json_path, verbose=True):
    if verbose:
        print("Reading %s" % progress_json_path)
    entries = dict()
    rows = []
    with open(progress_json_path, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            if len(line) > 0:
                row = json.loads(line)
                rows.append(row)
    all_keys = set(k for row in rows for k in row.keys())
    for k in all_keys:
        if k not in entries:
            entries[k] = []
        for row in rows:
            if k in row:
                v = row[k]
                try:
                    entries[k].append(float(v))
                except:
                    entries[k].append(np.nan)
            else:
                entries[k].append(np.nan)

        # entries[key] = [row.get(key, np.nan) for row in rows]
        #         added_keys = set()
        #         for k, v in row.items():
        #             if k not in entries:
        #                 entries[k] = []
        #             try:
        #                 entries[k].append(float(v))
        #             except:
        #                 entries[k].append(0.)
        #             added_keys.add(k)
        #         for k in entries.keys():
        #             if k not in added_keys:
        #                 entries[k].append(np.nan)
    entries = dict([(k, np.array(v)) for k, v in entries.items()])
    return entries


def flatten_dict(d):
    flat_params = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            v = flatten_dict(v)
            for subk, subv in flatten_dict(v).items():
                flat_params[k + "." + subk] = subv
        else:
            flat_params[k] = v
    return flat_params


def load_params(params_json_path):
    with open(params_json_path, 'r') as f:
        data = json.loads(f.read())
        if "args_data" in data:
            del data["args_data"]
        if "exp_name" not in data:
            data["exp_name"] = params_json_path.split("/")[-2]
    return data


def lookup(d, keys):
    if not isinstance(keys, list):
        keys = keys.split(".")
    for k in keys:
        if hasattr(d, "__getitem__"):
            if k in d:
                d = d[k]
            else:
                return None
        else:
            return None
    return d


def load_exps_data(exp_folder_paths, ignore_missing_keys=False, verbose=True):
    if isinstance(exp_folder_paths, str):
        exp_folder_paths = [exp_folder_paths]
    exps = []
    for exp_folder_path in exp_folder_paths:
        exps += [x[0] for x in os.walk(exp_folder_path)]
    if verbose:
        print("finished walking exp folders")
    exps_data = []
    for exp in exps:
        try:
            exp_path = exp
            variant_json_path = os.path.join(exp_path, "metadata.json")
            progress_json_path = os.path.join(exp_path, "progress.json")
            progress = load_progress(progress_json_path, verbose=verbose)
            try:
                params = load_params(variant_json_path)
            except IOError:
                params = dict(exp_name="experiment")
            exps_data.append(AttrDict(
                progress=progress, params=params, flat_params=flatten_dict(params)))
        except IOError as e:
            if verbose:
                print(e)

    # a dictionary of all keys and types of values
    all_keys = dict()
    for data in exps_data:
        for key in data.flat_params.keys():
            if key not in all_keys:
                all_keys[key] = type(data.flat_params[key])

    # if any data does not have some key, specify the value of it
    if not ignore_missing_keys:
        default_values = dict()
        for data in exps_data:
            for key in sorted(all_keys.keys()):
                if key not in data.flat_params:
                    if key not in default_values:
                        default = None
                        default_values[key] = default
                    data.flat_params[key] = default_values[key]

    return exps_data


def smart_repr(x):
    if isinstance(x, tuple):
        if len(x) == 0:
            return "tuple()"
        elif len(x) == 1:
            return "(%s,)" % smart_repr(x[0])
        else:
            return "(" + ",".join(map(smart_repr, x)) + ")"
    else:
        if hasattr(x, "__call__"):
            return "__import__('pydoc').locate('%s')" % (x.__module__ + "." + x.__name__)
        else:
            return repr(x)


def extract_distinct_params(exps_data, excluded_params=('exp_name', 'seed', 'log_dir'), l=1):
    try:
        stringified_pairs = sorted(
            map(
                eval,
                unique(
                    flatten(
                        [
                            list(
                                map(
                                    smart_repr,
                                    list(d.flat_params.items())
                                )
                            )
                            for d in exps_data
                        ]
                    )
                )
            ),
            key=lambda x: (
                tuple("" if it is None else str(it) for it in x),
            )
        )
    except Exception as e:
        print(e)
        import ipdb;
        ipdb.set_trace()
    proposals = [(k, [x[1] for x in v])
                 for k, v in itertools.groupby(stringified_pairs, lambda x: x[0])]
    filtered = [(k, v) for (k, v) in proposals if len(v) > l and all(
        [k.find(excluded_param) != 0 for excluded_param in excluded_params])]
    return filtered


class Selector(object):
    def __init__(self, exps_data, filters=None, custom_filters=None):
        self._exps_data = exps_data
        if filters is None:
            self._filters = tuple()
        else:
            self._filters = tuple(filters)
        if custom_filters is None:
            self._custom_filters = []
        else:
            self._custom_filters = custom_filters

    def where(self, k, v):
        return Selector(self._exps_data, self._filters + ((k, v),), self._custom_filters)

    def custom_filter(self, filter):
        return Selector(self._exps_data, self._filters, self._custom_filters + [filter])

    def _check_exp(self, exp):
        # or exp.flat_params.get(k, None) is None
        return all(
            ((str(exp.flat_params.get(k, None)) == str(v) or (k not in exp.flat_params)) for k, v in
             self._filters)
        ) and all(custom_filter(exp) for custom_filter in self._custom_filters)

    def extract(self):
        return list(filter(self._check_exp, self._exps_data))

    def iextract(self):
        return filter(self._check_exp, self._exps_data)


# Taken from plot.ly
color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]


def hex_to_rgb(hex, opacity=1.0):
    if hex[0] == '#':
        hex = hex[1:]
    assert (len(hex) == 6)
    return "rgba({0},{1},{2},{3})".format(int(hex[:2], 16), int(hex[2:4], 16), int(hex[4:6], 16), opacity)
