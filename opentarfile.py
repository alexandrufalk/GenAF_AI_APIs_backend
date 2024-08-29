import tarfile
import os

# Define the path to your .tar.gz file and the directory where you want to extract it
tar_gz_path = r'D:\Projects\GenAF_AI_APIs\genaf_ai_apis_backend\models\ObjectDetection\efficientdet-tensorflow2-d0-v1.tar.gz'
extract_to_dir = r'D:\Projects\GenAF_AI_APIs\genaf_ai_apis_backend\models\ObjectDetection\extracted_files'

# Ensure the extraction directory exists
os.makedirs(extract_to_dir, exist_ok=True)

# Open the .tar.gz file and extract its contents
with tarfile.open(tar_gz_path, 'r:gz') as tar:
    tar.extractall(path=extract_to_dir)

print(f'Extracted files to {extract_to_dir}')
