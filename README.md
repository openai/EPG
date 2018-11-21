**Status:** Archive (code is provided as-is, no updates expected)

# Evolved Policy Gradients (EPG)

The paper is located at https://arxiv.org/abs/1802.04821. A demonstration video can be found at https://youtu.be/-Z-ieH6w0LA.

> Houthooft, R., Chen, R. Y., Isola, P., Stadie, B. C., Wolski, F., Ho, J., Abbeel, P. (2018). Evolved Policy
Gradients. arXiv preprint arXiv:1802.04821.

### Installation

Install Anaconda:
```
curl -o /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash /tmp/miniconda.sh
conda create -n epg python=3.6.1
source activate epg
```

Install necessary OSX packages for MPI:
```
brew install open-mpi
```

Install necessary Python packages:
```
pip install mpi4py==3.0.0 scipy \
pandas tqdm joblib cloudpickle == 0.5.2 \
progressbar2 opencv-python flask >= 0.11.1 matplotlib pytest cython \
chainer pathos mujoco_py 'gym[all]'
```


### Running
First go to the EPG code folder:
```
cd <path_to_EPG_folder>
```
Then launch the entry script:
```
PYTHONPATH=. python epg/launch_local.py
```
Experiment data is saved in `<home_dir>/EPG_experiments/<month>-<day>/<experiment_name>`.

### Testing

First, set `theta_load_path = '<path_to_theta.npy>/theta.npy'` in `launch_local.py` according to the `theta.npy` obtained after running the `launch_local.py` script. This file should be located in `/<home_dir>/EPG_experiments/<month>-<day>/<experiment_name>/thetas/`.

Then run:
```
PYTHONPATH=. python epg/launch_local.py --test true
```

### Visualizing experiment data

Assuming the experiment data is saved in `<home_dir>/EPG_experiments/<month>-<day>/<experiment_name>`, run:
```
PYTHONPATH=. python epg/viskit/frontend.py <home_dir>/EPG_experiments/<month>-<day>/<experiment_name>
```
Then go to `http://0.0.0.0:5000` in your browser.

Viskit sourced from

> Duan, Y., Chen, X., Houthooft, R., Schulman, J., Abbeel, P. "Benchmarking Deep Reinforcement Learning for Continuous Control". Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016.

### BibTeX entry

```
@article{Houthooft18Evolved,
author = {Houthooft, Rein and Chen, Richard Y. and Isola, Phillip and Stadie, Bradly C. and Wolski, Filip and Ho, Jonathan and Abbeel, Pieter},
title = {Evolved Policy Gradients},
journal={arXiv preprint arXiv:1802.04821},
year = {2018}}
```