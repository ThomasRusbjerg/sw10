""" gsutil module to do common tasks with google cloud storage """
import os

from google.cloud import storage

def upload_blob(upload_dir, file_name):
    """ save blob to google storage (bucket) """

    # strip gs://
    upload_dir = upload_dir[5:]

    # split path to useful components
    path = upload_dir.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(path[0])

    dest = os.path.join(path[1], file_name)

    blob = bucket.blob(dest)
    blob.upload_from_filename(filename=file_name)

    print(
        "File {} uploaded to gs://{}.".format(
            file_name, upload_dir
        )
    )

def download_blob(download_dir, file_name, new_file_name=None):
    """ download blob from google storage (bucket) """

    # strip gs://
    download_dir = download_dir[5:]

    # split path to useful components
    path = download_dir.split("/", 1)
    # get model rom storage client
    client = storage.Client()
    bucket = client.bucket(path[0])
    src = os.path.join(path[1], file_name)

    # download the file, e.g. model, pca or scaler
    blob = bucket.blob(src)

    if new_file_name == None:
        blob.download_to_filename(filename=file_name)

        print(
            "File {} downloaded from gs://{}.".format(
                file_name, download_dir
            )
        )

    else:
        blob.download_to_filename(filename=new_file_name)
        print(
            "File {} downloaded from gs://{}.".format(
                new_file_name, download_dir
            )
        )
