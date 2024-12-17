import json
import pickle as pkl
import time
import os


def save_dct(fname, dict_name):
    if ".pkl" in fname:
        save_pkl(fname, dict_name)
    elif ".json" in fname:
        save_json(fname, dict_name)


def save_pkl(fname, dict_name):
    with open(fname, "wb") as f:
        pkl.dump(dict_name, f)


def save_json(fname, dict_name):
    with open(fname, "w") as f:
        f.write(json.dumps(dict_name, indent=4))


def load_dct(fname):
    if ".pkl" in fname:
        data = load_pkl(fname)
    elif ".json" in fname:
        data = load_json(fname)
    return data


def load_json(fname):
    data = json.loads(open(fname).read())
    return data


def load_pkl(fname, encdng=None):
    if encdng is None:
        with open(fname, "rb") as f:
            data = pkl.load(f)
    else:
        with open(fname, "rb") as f:
            data = pkl.load(f, encoding=encdng)
    return data


def write_to_file(write_str, fname):
    """Write string to file with detailed error logging"""
    try:
        print(f"\nAttempting to write to file: {fname}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        
        with open(fname, "w") as f:
            f.write(write_str)
        print(f"Successfully wrote to file: {fname}")
        
    except Exception as e:
        print(f"\nError in write_to_file:")
        print(f"File path: {fname}")
        raise  # Re-raise the exception after logging


def add_to_file(write_str, fname):
    with open(fname, "a") as f:
        f.write(write_str)


def read_lst(fname):
    with open(fname, "r") as f:
        data = [line.strip() for line in f.readlines()]
    return data


def format_time(start):
    current_sec = time.time() - start
    n_hrs = current_sec // 3600
    n_mins = (current_sec - (3600 * n_hrs)) // 60
    n_secs = current_sec - (3600 * n_hrs) - (60 * n_mins)

    time_str = "%d:%d:%d" % (n_hrs, n_mins, n_secs)
    return time_str
