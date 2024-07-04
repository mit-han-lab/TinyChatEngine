""" Python script to upload models to Hugging Face.

Usage:
   python tools/upload.py --filename <filename> --QM <method> --hf_token <token>

Example commandline:
   python tools/upload.py --filename LLaMA_3_8B_Instruct.zip --QM QM_ARM --hf_token <token>
"""
import argparse
import hashlib
import os
import zipfile

import requests
from tqdm import tqdm
from huggingface_hub import HfApi


def _upload_file_to_HF(filename, folder_name, hf_token):
    # Check if the file is a zip file
    if zipfile.is_zipfile(filename):
        print(f"Start uploading the model to Huggingface: mit-han-lab/tinychatengine-model-zoo/{folder_name}/{filename}")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=f"{folder_name}/{filename}",
            repo_id="mit-han-lab/tinychatengine-model-zoo",
            repo_type="model",
            commit_message="Upload models",
            token=hf_token
        )
        print(f"File uploaded successfully: mit-han-lab/tinychatengine-model-zoo/{folder_name}/{filename}")
    else:
        print(f"The file is not a zip file: {filename}")

def _remove_file(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)
        print(f"File removed successfully: {filepath}")
    else:
        print(f"Error: {filepath} not a valid filename")

def _main():
    parser = argparse.ArgumentParser(description="Download a file and check its md5sum")
    parser.add_argument("--filename", help="The name of the file to upload.")
    parser.add_argument("--QM", default="FP32", help="Quantization method.")
    parser.add_argument("--hf_token", help="Huggingface write token.")
    parser.add_argument("--remove_file", action="store_true", help="Remove the file after uploading.")
    args = parser.parse_args()

    Qmodels = ["FP32", "QM_ARM", "QM_x86", "QM_CUDA", "INT8"]

    if args.QM not in Qmodels:
        raise NotImplementedError(f"{args.QM} is not supported.")

    _upload_file_to_HF(args.filename, args.QM, args.hf_token) # Upload the file to Huggingface

    if args.remove_file:
        _remove_file(args.filename)  # Remove the zip file


if __name__ == "__main__":
    _main()
