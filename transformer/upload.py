"""Uploading models and asset to the dropbox storage.

Example commandline:
   python upload.py <dropbox app token>
"""
import argparse
import os

import dropbox

files_to_upload = [
    "assets.zip",
    "models.zip",
]


def subebackups(file_path, target_path, token):
    """Upload a file to the dropbox storage."""
    dbx = dropbox.Dropbox(token, timeout=36000)
    file_size = os.path.getsize(file_path)
    CHUNK_SIZE = 50 * 1024 * 1024
    dest_path = target_path

    with open(file_path, "rb") as f:
        if file_size <= CHUNK_SIZE:
            dbx.files_upload(f.read(), dest_path)

        else:
            upload_session_start_result = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
            cursor = dropbox.files.UploadSessionCursor(
                session_id=upload_session_start_result.session_id, offset=f.tell()
            )
            commit = dropbox.files.CommitInfo(path=dest_path, mode=dropbox.files.WriteMode("overwrite"))

            while f.tell() < file_size:
                if (file_size - f.tell()) <= CHUNK_SIZE:
                    print(dbx.files_upload_session_finish(f.read(CHUNK_SIZE), cursor, commit))
                else:
                    dbx.files_upload_session_append(f.read(CHUNK_SIZE), cursor.session_id, cursor.offset)
                    cursor.offset = f.tell()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a file to Dropbox.")
    parser.add_argument("token", help="Your Dropbox OAuth2 token.")
    args = parser.parse_args()

    db_prefix = "/MIT/transformer_assets/"
    local_prefix = "uploads"

    for file in files_to_upload:
        subebackups(file, db_prefix + file, args.token)
