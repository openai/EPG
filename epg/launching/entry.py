import argparse
import os.path

from epg.launching import launcher


def run(encoded_thunk, logdir):
    if os.path.isfile(encoded_thunk):
        with open(encoded_thunk, 'r') as f:
            encoded_thunk = f.read()
    thunk = launcher.decode_thunk(encoded_thunk)
    launcher.run_with_logger(thunk, logdir)


# Entrypoint for job
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('thunk')
    parser.add_argument('logdir')
    args = parser.parse_args()

    run(args.thunk, args.logdir)


if __name__ == '__main__':
    main()
