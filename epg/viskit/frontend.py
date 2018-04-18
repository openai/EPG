#!/usr/bin/env python

import argparse
import json
import os
import sys

import flask
import numpy as np
import plotly.graph_objs as go
import plotly.offline as po

from epg.viskit import core


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def unique(l):
    return list(set(l))


def flatten(l):
    return [item for sublist in l for item in sublist]


def sliding_mean(data_array, window=5):
    data_array = np.array(data_array)
    new_list = []
    for i in range(len(data_array)):
        indices = list(range(max(i - window + 1, 0),
                             min(i + window + 1, len(data_array))))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)


app = flask.Flask(__name__, static_url_path='/static')

exps_data = None
plottable_keys = None
x_plottable_keys = None
distinct_params = None
data_paths = None


@app.route('/js/<path:path>')
def send_js(path):
    return flask.send_from_directory('js', path)


@app.route('/css/<path:path>')
def send_css(path):
    return flask.send_from_directory('css', path)


def make_plot(plot_list, title=None):
    data = []
    for idx, plt in enumerate(plot_list):
        color = core.color_defaults[idx % len(core.color_defaults)]
        x = list(plt.xs)

        if plt.display_mode in ["mean_std", "mean_se"]:
            y = list(plt.means)
            if plt.display_mode == "mean_std":
                y_upper = list(plt.means + plt.stds)
                y_lower = list(plt.means - plt.stds)
            elif plt.display_mode == "mean_se":
                y_upper = list(plt.means + plt.ses)
                y_lower = list(plt.means - plt.ses)
            else:
                raise NotImplementedError
            data.append(go.Scatter(
                x=x + x[::-1],
                y=y_upper + y_lower[::-1],
                fill='tozerox',
                fillcolor=core.hex_to_rgb(color, 0.2),
                line=go.Line(color='transparent'),
                showlegend=False,
                legendgroup=plt.legend,
                hoverinfo='none'
            ))
            data.append(go.Scatter(
                x=x,
                y=y,
                name=plt.legend,
                legendgroup=plt.legend,
                line=dict(color=core.hex_to_rgb(color)),
            ))
        elif plt.display_mode == "individual":
            for idx, y in enumerate(plt.ys):
                data.append(go.Scatter(
                    x=x,
                    y=y,
                    name=plt.legend,
                    legendgroup=plt.legend,
                    line=dict(color=core.hex_to_rgb(color)),
                    showlegend=idx == 0,
                ))
        else:
            raise NotImplementedError

    layout = go.Layout(
        legend=dict(
            x=1,
            y=1,
        ),
        title=title,
    )
    fig = go.Figure(data=data, layout=layout)
    fig_div = po.plot(fig, output_type='div', include_plotlyjs=False)
    if "footnote" in plot_list[0]:
        footnote = "<br />".join([
            r"<span><b>%s</b></span>: <span>%s</span>" % (plt.legend, plt.footnote)
            for plt in plot_list
        ])
        return r"%s<div>%s</div>" % (fig_div, footnote)
    else:
        return fig_div


def summary_name(exp, selector=None):
    return exp.params["exp_name"]


def check_nan(exp):
    return all(not np.any(np.isnan(vals)) for vals in list(exp.progress.values()))


def get_plot_instruction(
        x_plot_key,
        plot_key,
        display_mode,
        split_key=None,
        group_key=None,
        filters=None,
):
    # print(x_plot_key, plot_key, split_key, group_key, filters)
    # if x_plot_key != "(default)":
    #     group_key = None
    selector = core.Selector(exps_data)
    if filters is None:
        filters = dict()
    for k, v in filters.items():
        selector = selector.where(k, str(v))

    if split_key is not None:
        vs = [vs for k, vs in distinct_params if k == split_key][0]
        split_selectors = [selector.where(split_key, v) for v in vs]
        split_legends = list(map(str, vs))
    else:
        split_selectors = [selector]
        split_legends = ["Plot"]
    plots = []
    counter = 1
    for split_selector, split_legend in zip(split_selectors, split_legends):
        if group_key and group_key is not "exp_name":
            vs = [vs for k, vs in distinct_params if k == group_key][0]
            group_selectors = [split_selector.where(group_key, v) for v in vs]
            group_legends = [str(x) for x in vs]
        else:
            group_key = "exp_name"
            vs = sorted([x.params["exp_name"] for x in split_selector.extract()])
            group_selectors = [split_selector.where(group_key, v) for v in vs]
            group_legends = [summary_name(x.extract()[0], split_selector) for x in group_selectors]

        to_plot = []
        for group_selector, group_legend in zip(group_selectors, group_legends):
            filtered_data = group_selector.extract()

            if len(filtered_data) > 0:

                progresses = [
                    exp.progress.get(plot_key, np.array([np.nan])) for exp in filtered_data]
                sizes = list(map(len, progresses))
                # more intelligent:
                max_size = max(sizes)
                progresses = [
                    np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses
                ]

                if x_plot_key == "(default)":
                    xs = np.arange(max_size)
                else:
                    # first decide what the xs should be
                    # ideally, it should be the union of

                    all_xs = np.unique(
                        np.sort(np.concatenate([d.progress.get(x_plot_key, []) for d in filtered_data])))
                    interp_progresses = []

                    for d in filtered_data:
                        if x_plot_key in d.progress:
                            assert plot_key in d.progress
                            interp_progresses.append(
                                np.interp(
                                    all_xs,
                                    d.progress[x_plot_key],
                                    d.progress[plot_key],
                                    right=np.nan
                                )
                            )
                        else:
                            continue

                    progresses = interp_progresses

                    xs = all_xs

                if display_mode == "mean_std":
                    means = np.nanmean(progresses, axis=0)
                    stds = np.nanstd(progresses, axis=0)
                    to_plot.append(AttrDict(
                        means=means,
                        stds=stds,
                        legend=group_legend,
                        xs=xs,
                        display_mode=display_mode,
                    ))
                elif display_mode == "mean_se":
                    means = np.nanmean(progresses, axis=0)
                    ses = np.nanstd(progresses, axis=0) / np.sqrt(np.sum(1 - np.isnan(progresses), axis=0))
                    to_plot.append(AttrDict(
                        means=means,
                        ses=ses,
                        legend=group_legend,
                        xs=xs,
                        display_mode=display_mode,
                    ))
                elif display_mode == "individual":
                    to_plot.append(AttrDict(
                        xs=xs,
                        ys=progresses,
                        legend=group_legend,
                        display_mode=display_mode,
                    ))
                else:
                    raise NotImplementedError

        if len(to_plot) > 0:
            fig_title = "%s: %s" % (split_key, split_legend)
            plots.append(make_plot(
                to_plot,
                title=fig_title,
            ))

        counter += 1
    return "\n".join(plots)


def parse_float_arg(args, key):
    x = args.get(key, "")
    try:
        return float(x)
    except Exception:
        return None


@app.route("/plot_div")
def plot_div():
    args = flask.request.args
    reload_s3 = args.get("reload_s3", False)
    x_plot_key = args.get("x_plot_key", "(default)")
    plot_key = args.get("plot_key")
    display_mode = args.get("display_mode", "mean_std")
    split_key = args.get("split_key", "")
    group_key = args.get("group_key", "")
    filters_json = args.get("filters", "{}")
    filters = json.loads(filters_json)
    if len(split_key) == 0:
        split_key = None
    if len(group_key) == 0:
        group_key = None

    print(reload_s3, type(reload_s3))

    if reload_s3:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        print(data_paths)
        for data_path in data_paths:
            if "data/s3/" in data_path:
                exp_group = data_path.split("data/s3/")[-1].split("/")[0]
                os.system("python %s/scripts/sync_s3.py %s" % (project_root, exp_group))
        reload_data()

    plot_div = get_plot_instruction(
        x_plot_key=x_plot_key,
        plot_key=plot_key,
        display_mode=display_mode,
        split_key=split_key,
        group_key=group_key,
        filters=filters,
    )
    return plot_div


@app.route("/")
def index():
    if "AverageReturn" in plottable_keys:
        plot_key = "AverageReturn"
    elif len(plottable_keys) > 0:
        plot_key = plottable_keys[0]
    else:
        plot_key = None
    if len(distinct_params) > 0:
        group_key = distinct_params[0][0]
    else:
        group_key = None
    print("Getting plot instruction...")
    plot_div = get_plot_instruction(
        x_plot_key="(default)", plot_key=plot_key, display_mode="mean_std", split_key=None,
        group_key=group_key)
    print("Rendering...")
    rendered = flask.render_template(
        "main.html",
        plot_div=plot_div,
        plot_key=plot_key,
        group_key=group_key,
        plottable_keys=plottable_keys,
        x_plot_key="(default)",
        x_plottable_keys=["(default)"] + x_plottable_keys,
        distinct_param_keys=[str(k) for k, v in distinct_params],
        distinct_params=dict([(str(k), list(map(str, v)))
                              for k, v in distinct_params]),
    )
    return rendered


def is_increasing_key(key, exps_data):
    for exp in exps_data:
        if key in exp.progress and not is_increasing(exp.progress[key]):
            return False
    return True


def is_increasing(arr):
    return np.all(arr[1:] - arr[:-1] >= 0) and np.max(arr) > np.min(arr)


def reload_data():
    global exps_data
    global plottable_keys
    global distinct_params
    global x_plottable_keys
    exps_data = core.load_exps_data(data_paths)
    plottable_keys = sorted(list(
        set(flatten(list(exp.progress.keys()) for exp in exps_data))))
    distinct_params = sorted(core.extract_distinct_params(exps_data))
    x_plottable_keys = [key for key in plottable_keys if is_increasing_key(key, exps_data)]


if __name__ == "__main__":
    default_port = int(os.environ.get("VISKIT_PORT", 5000))
    parser = argparse.ArgumentParser()
    parser.add_argument("data_paths", type=str, nargs='*')
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args(sys.argv[1:])

    data_paths = args.data_paths

    print("Importing data from {path}...".format(path=args.data_paths))
    reload_data()
    url = "http://localhost:%d" % (args.port)
    print("Done! View %s in your browser" % (url))

    app.run(host='0.0.0.0', port=args.port, debug=args.debug, threaded=True)
