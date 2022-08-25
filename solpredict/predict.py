# Copyright (c) Eli Lilly and Company and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Run solubility prediction under H6 condition given a full IgG sequence
import argparse
import os
import sys
import pathlib
import json
from pathlib import Path
import time
import torch
from model import MLP2Layer
from data import extract_seqid, organize_embed, organize_output


def create_parser():
    parser = argparse.ArgumentParser(
        description="Given a collection of IgG sequences, predict the solubility under H6 condition"
    )
    parser.add_argument(
        "--model_location",
        type=str,
        help="Directory of pretrained pytorch models, *pt format"
    )
    parser.add_argument(
        "--input_fasta",
        type=str,
        help="Sequence file in .fasta format"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory"
    )
    return parser

def main(args):
    timings = {}

    # Load fasta and organize seqid_ls
    seqid_ls = extract_seqid(args.input_fasta)

    # Extract esm1b embeddings
    output_dir_embed = os.path.join(args.output_dir, 'embed')
    Path(output_dir_embed).mkdir(parents=True, exist_ok=True)
    t_0 = time.time()
    cmd = f"python {os.path.join(os.path.dirname(__file__), 'extract.py')} esm1b_t33_650M_UR50S {args.input_fasta} {output_dir_embed} --repr_layers 33 --include mean per_tok"
    os.system(cmd)
    timings['embed'] = time.time() - t_0

    # Load esm1b embeddings
    embed_HC_LC_list = organize_embed(output_dir_embed, seqid_ls, EMB_LAYER=33)

    # Run the models for H6
    n_hidden_1 = 64
    n_hidden_2 = 32
    n_input = 2560
    n_output = 1
    predicted_dict = {}
    t_0 = time.time()
    for model_name in sorted(Path(args.model_location).glob('*_H6.pt')):
        model = MLP2Layer(n_input, n_hidden_1, n_hidden_2, n_output)
        model.load_state_dict(torch.load(os.path.join(args.model_location, model_name.name)))
        model.eval()
        predicted = model(torch.from_numpy(embed_HC_LC_list)).detach().numpy()
        predicted_dict[model_name.name] = predicted[:, 0]

   
    # Save the panda dataframe
    df_output = organize_output(predicted_dict, seqid_ls)
    df_output_name = os.path.join(args.output_dir, os.path.basename(args.input_fasta) + "_predicted_sol.csv")
    df_output.to_csv(df_output_name, index=False)

    # Save the timings
    timings['model compute'] = time.time() - t_0
    timings_output_path = os.path.join(args.output_dir, 'timings.json')
    with open(timings_output_path, 'w') as f:
        f.write(json.dumps(timings, indent=4))
    print("Finished")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)










