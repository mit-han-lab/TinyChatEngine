"""Python script to download model binaries.

Usage:
   python download_model.py --model <model_id> --QM <method>

Example commandline:
   python download_model.py --model LLaMA_7B_2_chat --QM QM_ARM
"""
import argparse
import hashlib
import os
import zipfile

import requests
from tqdm import tqdm

MODEL_DIR = "models"

# URLs and md5sums for models
models = {
    "LLaMA_7B_AWQ": {
        "url": "https://www.dropbox.com/s/c00cp1bcrwil8rl/LLaMA_7B_AWQ.zip",
        "md5sum": "0d2f13eadb4dd010102dbb473e909743",
    },
    "LLaMA_7B": {
        "url": "https://www.dropbox.com/scl/fi/xli03zmet80vxrib7tuq7/LLaMA_7B.zip?rlkey=98e2vvptmn6qz0fw2uudd3gb1&dl=1",  # noqa: E501
        "md5sum": "1ef6fde9b90745fa4c3d43e6d29d0c53",
    },
    "LLaMA_7B_2_chat": {
        "url": "https://www.dropbox.com/scl/fi/55hei425f6tiev9cc9h9r/LLaMA_7B_2_chat.zip?rlkey=42j8vp7d3crz6qm9om23vth25&dl=1",  # noqa: E501
        "md5sum": "fccbd388c1e89dd90e153ed6d9734189",
    },
    "OPT_6.7B": {
        "url": "https://www.dropbox.com/scl/fi/9ve3um8d63099qpx1h9ds/OPT_6.7B.zip?rlkey=9twzt8kpmttwbhj3nujkik0un&dl=1",
        "md5sum": "3fc742b690e32f2c51e59f96ebff3101",
    },
    "OPT_1.3B": {
        "url": "https://www.dropbox.com/scl/fi/qrmwcr5v4s4of9rj4jkve/OPT_1.3B.zip?rlkey=ye5qrt3wraypvebkkq69lg659&dl=1",
        "md5sum": "dc888e6ba784a948724bee1624d37a57",
    },
    "OPT_125m": {
        "url": "https://www.dropbox.com/scl/fi/9d6s8zd6nuzqyy7nf3j4h/OPT_125m.zip?rlkey=uqcg8tmdf6emps0l1eqy5xmcj&dl=1",
        "md5sum": "d106cee7300fcb980ac066026a8f3c8c",
    },
}

Qmodels = {
    "QM_ARM": {
        "LLaMA_7B_AWQ": {
            "url": "https://www.dropbox.com/scl/fi/y63qpsr5y2o2xka52wmem/LLaMA_7B_AWQ.zip?rlkey=dqnys72t6h27p64lx2bzcymjq&dl=1",  # noqa: E501
            "md5sum": "fd866ff3a2c4864318294b76274d10fe",
        },
        "LLaMA_7B": {
            "url": "https://www.dropbox.com/scl/fi/92yg27e0p21izx7lalryb/LLaMA_7B.zip?rlkey=97m442isg29es3ddon66nwfmy&dl=1",  # noqa: E501
            "md5sum": "4118dca49c39547929d9e02d67838153",
        },
        "LLaMA_7B_2_chat": {
            "url": "https://www.dropbox.com/scl/fi/1trpw92vmh4czvl28hkv0/LLaMA_7B_2_chat.zip?rlkey=dy1pdek0147gnuxdzpodi6pkt&dl=1",  # noqa: E501
            "md5sum": "af20c96de302c503a9fcfd5877ed0600",
        },
    },
    "QM_x86": {
        "LLaMA_7B_AWQ": {
            "url": "https://www.dropbox.com/scl/fi/qs2pk2gangeim1f7iafg7/LLaMA_7B_AWQ.zip?rlkey=ofe1mz4rz5xvgph9tk3s826vy&dl=1",  # noqa: E501
            "md5sum": "9f7750a5c58cdbf36fe9f20dffbba4b0",
        },
        "LLaMA_7B": {
            "url": "https://www.dropbox.com/scl/fi/i7yqzwr94wki2kywh9emr/LLaMA_7B.zip?rlkey=ce5j5p03wlwz5xdjrwuetxp4h&dl=1",  # noqa: E501
            "md5sum": "08c118ec34645808cd2d21678ad33659",
        },
        "LLaMA_7B_2_chat": {
            "url": "https://www.dropbox.com/scl/fi/vu7wnes1c7gkcegg854ys/LLaMA_7B_2_chat.zip?rlkey=q61o8fpc954g1ke6g2eaot7cf&dl=1",  # noqa: E501
            "md5sum": "18f2193ccb393c7bca328f42427ef233",
        },
    },
}


def _download_file(url, filepath):
    # Create a session
    with requests.Session() as session:
        session.headers = {"User-Agent": "Mozilla/5.0"}

        # Send a GET request to the URL
        response = session.get(url, stream=True)

        # Get the total file size
        file_size = int(response.headers.get("Content-Length", 0))

        # Check if the request was successful
        if response.status_code == 200:
            print(f"\nStart downloading the model to {filepath}.")
            # Download the file
            with open(filepath, "wb") as file:
                for chunk in tqdm(response.iter_content(chunk_size=1024), total=file_size // 1024, unit="KB"):
                    if chunk:
                        file.write(chunk)

            print(f"\nFile downloaded successfully: {filepath}")
        else:
            print(f"Failed to download the file. HTTP status code: {response.status_code}")


def _unzip_file(filepath, model_dir):
    print(f"Start unzip the model: {filepath}...")
    # Check if the file is a zip file
    if zipfile.is_zipfile(filepath):
        # Create a ZipFile object
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            # Extract all the contents of the zip file in the current directory
            zip_ref.extractall(model_dir)
            print(f"File unzipped successfully: {filepath}")
    else:
        print(f"The file is not a zip file: {filepath}")


def _remove_file(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)
        print(f"File removed successfully: {filepath}")
    else:
        print(f"Error: {filepath} not a valid filename")


def _md5(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _main():
    parser = argparse.ArgumentParser(description="Download a file and check its md5sum")
    parser.add_argument("--model", help="The name of the file to download.")
    parser.add_argument("--QM", default="fp32", help="Quantization method.")

    args = parser.parse_args()

    if args.model.startswith("LLaMA"):
        if args.QM == "fp32":
            model_table = models
            model_dir = MODEL_DIR
        elif args.QM in ["QM_ARM", "QM_x86"]:
            model_table = Qmodels[args.QM]
            model_dir = "."
        else:
            raise NotImplementedError(f"{args.QM} is not supported.")
    else:
        # OPT
        model_table = models
        model_dir = MODEL_DIR

    if args.model in model_table:
        url = model_table[args.model]["url"]
        expected_md5sum = model_table[args.model]["md5sum"]

        filepath = f"{model_dir}/{args.model}.zip"  # Save the file in the current directory
        _download_file(url, filepath)

        actual_md5sum = _md5(filepath)

        if actual_md5sum == expected_md5sum:
            print("The md5sum of the file matches the expected md5sum.")
            _unzip_file(filepath, model_dir)  # Unzip the file
            _remove_file(filepath)  # Remove the zip file
        else:
            print(
                "The md5sum of the file does not match the expected md5sum.",
                f"Expected: {expected_md5sum}, got: {actual_md5sum}",
            )
    else:
        raise ValueError(f"Unexpected model: {args.model}. Supported options: {models.keys()}")


if __name__ == "__main__":
    _main()
