import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive.file']
CACHE_FILE = 'upload_cache.json'

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {'folders': {}, 'files': []}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)



def create_drive_folder(service, folder_name, parent_id=None):
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id] if parent_id else []
    }
    folder = service.files().create(body=file_metadata, fields='id').execute()
    return folder.get('id')

def upload_file(service, file_path, parent_id):
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [parent_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    print(f"Uploading file with size {os.path.getsize(file_path)/1024/1024/1024:.2f} GB: {file_path}")
    # Check if the file is already uploaded
    service.files().create(body=file_metadata, media_body=media, fields='id').execute()

def upload_folder(service, folder_path, parent_folder_id=None):
    cache = load_cache()
    folder_name = os.path.basename(folder_path)

    # Create root folder if not cached
    if folder_path not in cache['folders']:
        root_drive_folder_id = create_drive_folder(service, folder_name, parent_folder_id)
        cache['folders'][folder_path] = root_drive_folder_id
        save_cache(cache)
    else:
        root_drive_folder_id = cache['folders'][folder_path]

    path_to_drive_id = {folder_path: root_drive_folder_id}

    for current_path, dirs, files in os.walk(folder_path):
        print(f"Uploading {current_path}...")
        for dir_name in dirs:
            local_dir_path = os.path.join(current_path, dir_name)
            if local_dir_path not in cache['folders']:
                parent_drive_id = path_to_drive_id[current_path]
                drive_folder_id = create_drive_folder(service, dir_name, parent_drive_id)
                cache['folders'][local_dir_path] = drive_folder_id
                save_cache(cache)
            else:
                drive_folder_id = cache['folders'][local_dir_path]
            path_to_drive_id[local_dir_path] = drive_folder_id

        for file_name in files:
            file_path = os.path.join(current_path, file_name)
            if file_path in cache['files']:
                print(f"Skipping already uploaded file: {file_path}")
                continue
            parent_drive_id = path_to_drive_id[current_path]
            try:
                upload_file(service, file_path, parent_drive_id)
                cache['files'].append(file_path)
                save_cache(cache)
                print(f"Uploaded file: {file_path}")
            except Exception as e:
                print(f"Failed to upload {file_path}: {e}")
                # Don't stop the whole script if one file fails

def main():
    service = authenticate()
    # folder_path = input("Enter the local folder path you want to upload: ").strip()
    
    # if not os.path.isdir(folder_path):
    #     print("Invalid folder path. Please check and try again.")
    #     return
    # print("Uploading folder...")
    # upload_folder(service, folder_path)
    # print("Upload completed.")
    # '/scratch/dr/n.eddine/crop_classification_tifs/data_window_approach_2x2/Train/train.csv','/scratch/dr/n.eddine/crop_classification_tifs/window approach 2x2_others/ghana_parquets/train_others.parquet' \
    # /scratch/dr/n.eddine/crop_classification_tifs/window_approach_2x2_new_split_val_test/valid/valid.parquet,/scratch/dr/n.eddine/crop_classification_tifs/window approach 2x2_others/ghana_parquets/val_others.parquet' \

    files = ['/scratch/dr/n.eddine/crop_classification_tifs/data_window_approach_2x2/Train/train.csv',
            '/scratch/dr/n.eddine/crop_classification_tifs/window approach 2x2_others/ghana_parquets/train_others.parquet',
            '/scratch/dr/n.eddine/crop_classification_tifs/window_approach_2x2_new_split_val_test/valid/valid.parquet',
            '/scratch/dr/n.eddine/crop_classification_tifs/window approach 2x2_others/ghana_parquets/val_others.parquet']
    drive_folder_name = 'data_crop'
    print("Uploading files...")
    parent_id = '1yyka1mTW3IlUrf7Ekt45QN_DxZW-dYiF'
    for file in files:
        upload_file(service, file, parent_id)
        print(f"Uploaded file: {file}")
    print("Upload completed.")
if __name__ == '__main__':
    main()
