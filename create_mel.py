from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import json
import torch
import numpy as np
from env import AttrDict
from meldataset import spectrogram_to_mel
from tqdm import tqdm


def get_mel(x):
    return spectrogram_to_mel(
        x,
        h.n_fft,
        h.num_mels,
        h.sampling_rate,
        h.fmin,
        h.fmax,
    )


def create_mel(a):
    filelist = [f for f in os.listdir(a.input_dir) if f.find(".npy") != -1]

    os.makedirs(a.output_dir, exist_ok=True)

    with torch.no_grad():
        for filname in tqdm(filelist):
            spec = np.load(os.path.join(a.input_dir, filname))
            spec = torch.from_numpy(spec).to(device)
            x = get_mel(spec)
            np.save(os.path.join(a.out_dir, filname), x.cpu())


def main():
    print("Initializing Create Mel Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="test_files")
    parser.add_argument("--output_dir", default="generated_files")
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--mode", default="wav", choices=["wav", "spec"])
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], "config.json")
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    create_mel(a)


if __name__ == "__main__":
    main()
