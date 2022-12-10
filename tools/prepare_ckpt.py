# This script download pretrained checkpoint from google drive and save it to local disk.
import os


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


if __name__ == "__main__":
    # download checkpoint from google drive
    file_id = "1rJ8WdY6"
    download_file_from_google_drive(id, destination)
