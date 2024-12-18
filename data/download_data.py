import os
import gdown

def download_file_from_google_drive(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Define file IDs and destination paths
    files_to_download = [
        {
            'file_id': '1vK1flNSOMADCmYnxQjSHoq3FupZ42Anq',
            'dest_path': 'data/X_unlabeled.npy'
        },
        {
            'file_id': '1sXJX-1Tge5voIPXxFhMene1HLKhfp9gm',
            'dest_path': 'data/X.npy'
        },
        {
            'file_id': '1jgNQ2M5mrtjqo7V_fCpHbGigslc1jjOC',
            'dest_path': 'data/y.npy'
        },
        {
            'file_id': '1ZXSvlVzxipCP_EDsUMtPE4TM0ygVMxaJ',
            'dest_path': 'data/sub.npy'
        },
    ]

    # Download each file
    for file_info in files_to_download:
        print(f"Downloading {file_info['dest_path']}...")
        download_file_from_google_drive(file_info['file_id'], file_info['dest_path'])
        print(f"Downloaded {file_info['dest_path']}.\n")
