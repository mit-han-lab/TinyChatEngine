"""Python script to download model binaries.

Usage:
   python download_model.py --model <model_id> --QM <method>

Example commandline:
   python download_model.py --model LLaMA_3_8B_Instruct_awq_int4 --QM QM_ARM
"""
import argparse
import hashlib
import os
import zipfile

import requests
from tqdm import tqdm

from huggingface_hub import hf_hub_download

MODEL_DIR = "."

# URLs and md5sums for models
models = {
    "LLaMA_7B_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/LLaMA_7B.zip?download=true",  # noqa: E501
        "md5sum": "1ef6fde9b90745fa4c3d43e6d29d0c53",
    },
    "LLaMA2_7B_chat_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/LLaMA_7B_2_chat.zip?download=true",  # noqa: E501
        "md5sum": "fccbd388c1e89dd90e153ed6d9734189",
    },
    "LLaMA2_13B_chat_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/LLaMA_13B_2_chat.zip?download=true",  # noqa: E501
        "md5sum": "59b73efa638be4131e5fd27c3fdee597",
    },
    "CodeLLaMA_7B_Instruct_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/CodeLLaMA_7B_Instruct.zip?download=true",
        "md5sum": "6ca1682140cd8f91316a8b5b7dee6cd4",
    },
    "CodeLLaMA_13B_Instruct_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/CodeLLaMA_13B_Instruct.zip?download=true",
        "md5sum": "ebcb20dec3bd95a7e5a6eb93ed4c178d",
    },
    "opt_6.7B_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/OPT_6.7B.zip?download=true",
        "md5sum": "69cffdc090388ac2d2abcbe8163b0397",
    },
    "opt_1.3B_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/OPT_1.3B.zip?download=true",
        "md5sum": "a49490989fe030a0e9dee119285f7cf5",
    },
    "opt_125m_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/OPT_125m.zip?download=true",
        "md5sum": "816958aed84120b763942ba83c1b010f",
    },
    "LLaVA_7B_CLIP_ViT-L_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/LLaVA_7B_CLIP_ViT-L.zip?download=true",  # noqa: E501
        "md5sum": "dc4df9a17c7810333a6b9d561f4b8218",
    },
    "LLaVA_13B_CLIP_ViT-L_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/LLaVA_13B_CLIP_ViT-L.zip?download=true",  # noqa: E501
        "md5sum": "3d4afd8051c779c014ba69aec7886961",
    },
    "VILA_2.7B_CLIP_ViT-L_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/VILA_2.7B_CLIP_ViT-L.zip?download=true",  # noqa: E501
        "md5sum": "48455c57594ea1a6b44496fda3877c75",
    },
    "VILA_7B_CLIP_ViT-L_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/VILA_7B_CLIP_ViT-L.zip?download=true",  # noqa: E501
        "md5sum": "d2201fd2853da56c3e2b4b7043b1d37a",
    },
    "StarCoder_15.5B_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/StarCoder_15.5B.zip?download=true",
        "md5sum": "e3e9301866f47ab84817b46467ac49f6",
    },
    "Mistral_7B_v0.2_Instruct_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/Mistral_7B_v0.2_Instruct.zip?download=true",
        "md5sum": "8daa04f2af5f0470c66eb45615ab07e2",
    },
    "LLaMA_3_8B_Instruct_fp32": {
        "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/FP32/LLaMA_3_8B_Instruct.zip?download=true",
        "md5sum": "ae710f37a74f98d0d47da085672179b5",
    },
    "VILA1.5_8B_fp32": {
        "url": "",  # noqa: E501
        "md5sum": "",
    },
}

Qmodels = {
    "QM_ARM": {
        "LLaMA_7B_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/LLaMA_7B.zip?download=true",  # noqa: E501
            "md5sum": "4118dca49c39547929d9e02d67838153",
        },
        "LLaMA2_7B_chat_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/LLaMA_7B_2_chat.zip?download=true",  # noqa: E501
            "md5sum": "af20c96de302c503a9fcfd5877ed0600",
        },
        "LLaMA2_13B_chat_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/LLaMA_13B_2_chat.zip?download=true",  # noqa: E501
            "md5sum": "f1f7693da630bb7aa269ecae5bcc397a",
        },
        "CodeLLaMA_7B_Instruct_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/CodeLLaMA_7B_Instruct.zip?download=true",
            "md5sum": "a5b4c15857944daaa1e1ee34c5917264",
        },
        "CodeLLaMA_13B_Instruct_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/CodeLLaMA_13B_Instruct.zip?download=true",
            "md5sum": "d749ec83a54dcf40a7d87e7dbfba42d4",
        },
        "opt_125m_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/OPT_125m.zip?download=true",  # noqa: E501
            "md5sum": "2b42c3866c54642557046140367217fa",
        },
        "opt_1.3B_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/OPT_1.3B.zip?download=true",  # noqa: E501
            "md5sum": "1fb1296184c8c61e4066775ba59573b9",
        },
        "opt_6.7B_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/OPT_6.7B.zip?download=true",  # noqa: E501
            "md5sum": "6d061dc64ccc60864391f484b5e564d0",
        },
        "LLaVA_7B_awq_int4_CLIP_ViT-L": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/LLaVA_7B_CLIP_ViT-L.zip?download=true",  # noqa: E501
            "md5sum": "9fa1bc2f8c9b06b46c1f37bd2b17702c",
        },
        "LLaVA_13B_awq_int4_CLIP_ViT-L": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/LLaVA_13B_CLIP_ViT-L.zip?download=true",  # noqa: E501
            "md5sum": "fec078d99449df73c0f1236377b53eb3",
        },
        "VILA_2.7B_awq_int4_CLIP_ViT-L": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/VILA_2.7B_CLIP_ViT-L.zip?download=true",
            "md5sum": "177b1a58707355c641da4f15fb3c7a71",
        },
        "VILA_7B_awq_int4_CLIP_ViT-L": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/VILA_7B_CLIP_ViT-L.zip?download=true",  # noqa: E501
            "md5sum": "29aa8688b59dfde21d0b0b0b94b0ac27",
        },
        "StarCoder_15.5B_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/StarCoder_15.5B.zip?download=true",
            "md5sum": "0f16236c0aec0b32b553248cc78b8caf",
        },
        "Mistral_7B_v0.2_Instruct_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/Mistral_7B_v0.2_Instruct.zip?download=true",
            "md5sum": "ac897d408a702ae79252bc79bfbbb699",
        },
        "LLaMA_3_8B_Instruct_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/LLaMA_3_8B_Instruct.zip?download=true",
            "md5sum": "8c44a5d7cb2a0406f8f1cbb785ed7e17",
        },
        "VILA1.5_8B_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_ARM/VILA1.5_8B.zip?download=true",  # noqa: E501
            "md5sum": "9e9ab4e30f9fc7de69fadb3aae511456",
        },
    },
    "QM_x86": {
        "LLaMA_7B_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/LLaMA_7B.zip?download=true",  # noqa: E501
            "md5sum": "08c118ec34645808cd2d21678ad33659",
        },
        "LLaMA2_7B_chat_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/LLaMA_7B_2_chat.zip?download=true",  # noqa: E501
            "md5sum": "18f2193ccb393c7bca328f42427ef233",
        },
        "LLaMA2_13B_chat_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/LLaMA_13B_2_chat.zip?download=true",  # noqa: E501
            "md5sum": "3684e5740f44ed05e213d6d807a1f136",
        },
        "CodeLLaMA_7B_Instruct_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/CodeLLaMA_7B_Instruct.zip?download=true",
            "md5sum": "b208eec1b1bbb6532f26b68a7a3caae6",
        },
        "CodeLLaMA_13B_Instruct_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/CodeLLaMA_13B_Instruct.zip?download=true",
            "md5sum": "71ade74fe50b6beb378d52e19396926d",
        },
        "opt_125m_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/OPT_125m.zip?download=true",  # noqa: E501
            "md5sum": "c9c26bb5c8bf9867e21e525da744ef19",
        },
        "opt_1.3B_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/OPT_1.3B.zip?download=true",  # noqa: E501
            "md5sum": "dd4801d7b65915a70a29d1d304ce5783",
        },
        "opt_6.7B_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/OPT_6.7B.zip?download=true",  # noqa: E501
            "md5sum": "4aba1bee864029d06d1fec67f4d95a22",
        },
        "LLaVA_7B_awq_int4_CLIP_ViT-L": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/LLaVA_7B_CLIP_ViT-L.zip?download=true",  # noqa: E501
            "md5sum": "f903927fe3d02d9db7fb8f0c6587c136",
        },
        "LLaVA_13B_awq_int4_CLIP_ViT-L": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/LLaVA_13B_CLIP_ViT-L.zip?download=true",  # noqa: E501
            "md5sum": "f22e8d5d754c64f0aa34d5531d3059bc",
        },
        "VILA_2.7B_awq_int4_CLIP_ViT-L": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/VILA_2.7B_CLIP_ViT-L.zip?download=true",
            "md5sum": "e83ff23d58a0b91c732a9e3928aa344a",
        },
        "VILA_7B_awq_int4_CLIP_ViT-L": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/VILA_7B_CLIP_ViT-L.zip?download=true",  # noqa: E501
            "md5sum": "7af675198ec3c73d440ccc96b2722813",
        },
        "StarCoder_15.5B_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/StarCoder_15.5B.zip?download=true",
            "md5sum": "48383ce0bf01b137069e3612cab8525f",
        },
        "Mistral_7B_v0.2_Instruct_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/Mistral_7B_v0.2_Instruct.zip?download=true",
            "md5sum": "66f24d7ca1e12f573e172d608536f997",
        },
        "LLaMA_3_8B_Instruct_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/LLaMA_3_8B_Instruct.zip?download=true",
            "md5sum": "8540fec0fefa44e13e81748ff8edb231",
        },
        "VILA1.5_8B_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_x86/VILA1.5_8B.zip?download=true",  # noqa: E501
            "md5sum": "1c0574fa1d4aa81616a655bc3436479c",
        },
    },
    "QM_CUDA": {
        "LLaMA2_7B_chat_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_CUDA/LLaMA_7B_2_chat.zip?download=true",  # noqa: E501
            "md5sum": "fedc72e60962128e3dbf070f770c5825",
        },
        "LLaMA2_13B_chat_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_CUDA/LLaMA_13B_2_chat.zip?download=true",  # noqa: E501
            "md5sum": "24fd0af09f68d260b73594266ef1ee5d",
        },
        "CodeLLaMA_7B_Instruct_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_CUDA/CodeLLaMA_7B_Instruct.zip?download=true",
            "md5sum": "c6bf03ddb47a7cbf1dc370fac8250c90",
        },
        "CodeLLaMA_13B_Instruct_awq_int4": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/QM_CUDA/CodeLLaMA_13B_Instruct.zip?download=true",
            "md5sum": "3a9c5d2ed1863e686eba98221a618820",
        },
    },
    "INT8": {
        "opt_125m_smooth_int8": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/INT8/OPT_125m.zip?download=true",  # noqa: E501
            "md5sum": "e3bf0b7f13f393aa054de00a8433f232",
        },
        "opt_1.3B_smooth_int8": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/INT8/OPT_1.3B.zip?download=true",  # noqa: E501
            "md5sum": "d40d797f32d7b8a0a8648e9395575b12",
        },
        "opt_6.7B_smooth_int8": {
            "url": "https://huggingface.co/mit-han-lab/tinychatengine-model-zoo/resolve/main/INT8/OPT_6.7B.zip?download=true",  # noqa: E501
            "md5sum": "635994471bc6e56857cfa75f9949a090",
        },
    },
}


def _download_file(url, filepath, QM):
    if QM == "FP32":
        folder_name = "FP32"
    else:
        folder_name = QM

    print(f"\nStart downloading the model to {filepath}.")
    hf_hub_download(repo_id="mit-han-lab/tinychatengine-model-zoo", 
                    subfolder=folder_name,
                    filename=filepath,
                    force_download=True,
                    local_dir=".")
    print(f"\nFile downloaded successfully: {filepath}")


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
    parser.add_argument("--QM", default="FP32", help="Quantization method.")

    args = parser.parse_args()

    if args.QM == "FP32":
        model_table = models
        model_dir = MODEL_DIR
    elif args.QM in Qmodels:
        model_table = Qmodels[args.QM]
        model_dir = "."
    else:
        raise NotImplementedError(f"{args.QM} is not supported.")

    if args.model in model_table:
        url = model_table[args.model]["url"]
        expected_md5sum = model_table[args.model]["md5sum"]

        # filepath = f"{model_dir}/{args.model}.zip"  # Save the file in the current directory
        filepath = f"{model_dir}/{url.split('/')[-1].split('?')[0]}"  # Save the file in the current directory
        # if the file exists, delete it
        if os.path.isfile(filepath):
            print(f"File already exists: {filepath}. Removing it.")
            os.remove(filepath)
        _download_file(url, filepath, args.QM)

        # Move the file from args.QM folder to the current directory
        os.rename(f"{args.QM}/{filepath}", f"./{filepath}")
        os.rmdir(args.QM)

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
