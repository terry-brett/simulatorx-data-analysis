import requests

AGE_MODEL_ID = "1-NnO9GiRVAvugq4Oc5XE2Ezoj596oB56"
ETHNICITY_MODEL_ID = "1-Dz-kjs2ny5pRshXzfrA-7tWvPNm3mN6"
GENDER_MODEL_ID = "1-VfamgvLQf1ClHfhia5oMMvK19ICTDSD"

def download_file_from_google_drive(id, destination, file):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    print ("Downloading...", file)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download(file, path):
    ID = None
    if 'age' in file:
        ID = AGE_MODEL_ID
    if 'eth' in file:
        ID = ETHNICITY_MODEL_ID
    if 'gen' in file:
        ID = GENDER_MODEL_ID

    destination = path + "/" + file
    if ID is not None:
        download_file_from_google_drive(ID, destination, file)
    else:
        print ("Something went wrong")