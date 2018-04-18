import os

import numpy as np

from epg.launching import logger
from epg.utils import ret_to_obj


def plot_results(itr, results):
    import matplotlib.pyplot as plt

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

    f, axarr = plt.subplots(2, len(results), figsize=(24, 6))
    f.tight_layout()
    for idx, r in enumerate(results):
        smoothed_ret = sliding_mean(r['ep_return'], window=np.maximum(int(len(r['ep_return']) / 50), 1))
        axarr[0, idx].plot(range(len(smoothed_ret)), smoothed_ret, linewidth=1.0, color='red')
        obj = ret_to_obj(r['ep_return'])
        axarr[0, idx].set_title('{:.3f}'.format(obj), y=0.8)
        axarr[1, idx].plot(range(len(r['ep_kl'])), r['ep_kl'], linewidth=1.0, color='blue')
    plt.show()
    save_path = os.path.join(logger.get_dir(), 'analysis')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'epoch_{}.png'.format(itr)))
    plt.clf()
    plt.close()
