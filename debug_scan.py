import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(BASE_DIR, "dataset_raw")

print(f"Base Dir: {BASE_DIR}")
print(f"Raw Dir: {raw_dir}")

if not os.path.exists(raw_dir):
    print("Raw dir does not exist!")
else:
    print("Raw dir exists.")
    print(f"Contents of raw_dir: {os.listdir(raw_dir)}")
    
    for item in os.listdir(raw_dir):
        item_path = os.path.join(raw_dir, item)
        if os.path.isdir(item_path):
            print(f"Scanning subfolder: {item}")
            for root, dirs, files in os.walk(item_path):
                print(f"  Root: {root}")
                print(f"  Files: {files}")
                for f in files:
                    if f.lower().endswith(('.wav', '.flac', '.mp3', '.m4a', '.ogg')):
                        print(f"    Found audio: {f}")
