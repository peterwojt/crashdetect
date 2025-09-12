import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration ---
source_folder = 'videos'  # Where your original files are
csv_path = 'labels.csv'   # Path to CSV file
destination_root = 'crash_no_crash_dataset'  # Root folder for crash / no_crash

# --- Load CSV ---
df = pd.read_csv(csv_path)

# --- Ensure destination folders exist ---
for category in ['crash', 'no_crash']:
    for subset in ['train', 'val', 'test']:
        # Skip crash/train folder
        if category == 'crash' and subset == 'train':
            continue
        os.makedirs(os.path.join(destination_root, category, subset), exist_ok=True)

# --- Separate crash and no_crash entries ---
df_crash = df[df['label'] == 'c']
df_no_crash = df[df['label'].isin(['n', 'z'])]

# --- Split crash into 50% val, 50% test ---
df_crash_val, df_crash_test = train_test_split(df_crash, test_size=0.5, random_state=42)

# --- Split no_crash into 70% train, 15% val, 15% test ---
df_no_crash_train, df_no_crash_temp = train_test_split(df_no_crash, test_size=0.30, random_state=42)
df_no_crash_val, df_no_crash_test = train_test_split(df_no_crash_temp, test_size=0.5, random_state=42)

# --- Copy files ---
def copy_files(df_subset, destination_subfolder):
    for _, row in df_subset.iterrows():
        src_file = os.path.join(source_folder, row['filename'])
        dest_file = os.path.join(destination_subfolder, row['filename'])
        if os.path.exists(src_file):
            shutil.copy2(src_file, dest_file)
        else:
            print(f"Warning: {src_file} not found.")

# --- Copy crash files (val and test only) ---
copy_files(df_crash_val, os.path.join(destination_root, 'crash', 'val'))
copy_files(df_crash_test, os.path.join(destination_root, 'crash', 'test'))

# --- Copy no_crash files (train, val, test) ---
copy_files(df_no_crash_train, os.path.join(destination_root, 'no_crash', 'train'))
copy_files(df_no_crash_val, os.path.join(destination_root, 'no_crash', 'val'))
copy_files(df_no_crash_test, os.path.join(destination_root, 'no_crash', 'test'))

print("File distribution complete.")
