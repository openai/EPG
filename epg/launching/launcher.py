import base64
import json
import os
import pickle
import shutil
import sys
import traceback
import zlib

import cloudpickle

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

LOCAL_LOG_PATH = os.path.expanduser("~/EPG_experiments")


def dumps_with_help(obj):
    if cloudpickle.__version__ != '0.5.2':
        raise RuntimeError(
            'cloudpickle version 0.5.2 is required, please run `pip install cloudpickle==0.5.2`')
    try:
        return cloudpickle.dumps(obj)
    except Exception:
        raise RuntimeError(
            'Failed to cloudpickle %s. Possible fixes: (1) remove super() from yours script.' % obj)


def encode_thunk(thunk):
    return sys.version[:3] + base64.b64encode(zlib.compress(dumps_with_help(thunk))).decode('utf-8')


def decode_thunk(encoded_thunk):
    local_version = encoded_thunk[:3]
    actual_thunk = encoded_thunk[3:]
    remote_version = sys.version[:3]
    assert local_version[:3] == remote_version[:3], \
        'Version mismatch! Local machine uses python %s,' \
        ' remote machine uses python %s' % (local_version, remote_version)
    return cloudpickle.loads(zlib.decompress(base64.b64decode(actual_thunk)))


def write_metadata(dir, args, kwargs):
    with open(os.path.join(dir, 'metadata.json'), 'wt') as fh:
        fh.write(json.dumps(dict(args=args, kwargs=kwargs)))


def colorize(string, color='green', bold=False, highlight=False):
    """Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    attrs = ';'.join(attr)
    return '\x1b[%sm%s\x1b[0m' % (attrs, string)


def make_command(thunk, logdir, mpi_num_procs, mpi_hosts_path):
    b64thunk = encode_thunk(thunk)
    cmd = ['python', '-u', '-m', 'epg.launching.entry', b64thunk, logdir]
    if mpi_num_procs > 1:
        mpicmd = ['mpirun', '-np', str(mpi_num_procs)]
        if mpi_hosts_path:
            mpicmd.extend(['-f', mpi_hosts_path])
        mpicmd.extend(cmd)
        cmd = mpicmd
    return cmd


def atomic_write(bytes, filename):
    with open(filename + '.tmp', 'wb') as fh:
        fh.write(bytes)
    shutil.move(filename + '.tmp', filename)


def run_with_logger(thunk, logdir):
    from epg.launching import logger
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        os.makedirs(logdir, exist_ok=True)
    try:
        with logger.scoped_configure(dir=logdir, format_strs=None if rank == 0 else []):
            retval = thunk()
            if rank == 0:
                atomic_write(pickle.dumps(retval, protocol=-1), os.path.join(logdir, 'retval.pkl'))
            return retval
    except Exception as e:
        with open(os.path.join(logdir, "exception%i.txt" % rank), 'wt') as fh:
            fh.write(traceback.format_exc())
        raise e


def call(fn, *, log_relpath, args=None, kwargs=None, mpi_proc_per_machine=1, mpi_machines=1, **__):
    local_eval_dir = os.path.join(LOCAL_LOG_PATH, log_relpath)
    if os.path.exists(local_eval_dir):
        print(colorize(
            'Directory %s exists. Removing existing data (this is the default behavior for backend=local)' % local_eval_dir,
            color='red', highlight=True))
        shutil.rmtree(local_eval_dir)

    os.makedirs(local_eval_dir, exist_ok=True)
    args = args or []
    kwargs = kwargs or {}

    write_metadata(local_eval_dir, args=args, kwargs=kwargs)

    def thunk():
        return fn(*args, **kwargs)

    mpi_procs = mpi_proc_per_machine * mpi_machines
    if mpi_procs > 1:
        cmd = make_command(thunk, local_eval_dir, mpi_procs, mpi_hosts_path=None)
        return os.execvp(cmd[0], cmd)
    else:
        return run_with_logger(thunk, local_eval_dir)
