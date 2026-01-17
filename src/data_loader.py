import os
import urllib.request
import zipfile
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DATA_FILE_PATH = os.path.join(DATA_DIR, 'SMSSpamCollection')

def download_data():
    """Download and extract the SMS Spam Collection dataset."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    zip_path = os.path.join(DATA_DIR, 'smsspamcollection.zip')
    
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Downloading data from {DATA_URL}...")
        try:
            urllib.request.urlretrieve(DATA_URL, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None
    else:
        print("Data already exists.")
    
    return DATA_FILE_PATH


def load_youtube_spam_data():
    """Download and load the YouTube Spam Collection dataset."""
    YOUTUBE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip"
    zip_path = os.path.join(DATA_DIR, 'youtube_spam.zip')
    extract_dir = os.path.join(DATA_DIR, 'youtube_spam')

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if not os.path.exists(extract_dir):
        print(f"Downloading YouTube Spam Data from {YOUTUBE_URL}...")
        try:
            urllib.request.urlretrieve(YOUTUBE_URL, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("YouTube Download complete.")
        except Exception as e:
            print(f"Error downloading YouTube data: {e}")
            return None
            
    # Load all CSVs in the directory
    dfs = []
    for filename in os.listdir(extract_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(extract_dir, filename)
            # YouTube dataset has headers: COMMENT_ID, AUTHOR, DATE, CONTENT, CLASS
            # CLASS = 1 for spam, 0 for ham
            try:
                temp_df = pd.read_csv(file_path)
                temp_df = temp_df[['CONTENT', 'CLASS']]
                temp_df.columns = ['message', 'label_blo']
                # Map 1->spam, 0->ham
                temp_df['label'] = temp_df['label_blo'].map({1: 'spam', 0: 'ham'})
                dfs.append(temp_df[['label', 'message']])
            except Exception as e:
                print(f"Skipping {filename}: {e}")
                
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

def load_data():
    """Load and combine SMS and YouTube datasets."""
    print("Loading SMS Spam Collection...")
    sms_file_path = download_data()
    sms_df = None
    if sms_file_path and os.path.exists(sms_file_path):
        sms_df = pd.read_csv(sms_file_path, sep='\t', header=None, names=['label', 'message'], quoting=3)
        
    print("Loading YouTube Spam Collection...")
    yt_df = load_youtube_spam_data()
    
    if sms_df is not None and yt_df is not None:
        print(f"Combined SMS ({len(sms_df)}) and YouTube ({len(yt_df)}) data.")
        return pd.concat([sms_df, yt_df], ignore_index=True)
    elif sms_df is not None:
        return sms_df
    elif yt_df is not None:
        return yt_df
    else:
        print("Failed to load any data.")
        return None

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(df.head())
        print(df['label'].value_counts())
