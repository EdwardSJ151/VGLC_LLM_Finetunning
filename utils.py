import os
from typing import List

def find_txt_files(directories: List[str]) -> List[str]:
    """Finds all .txt files within the specified directories."""
    txt_files = []
    print("Searching for .txt files in:")
    for directory in directories:
        print(f"- {directory}")
        if not os.path.isdir(directory):
            print(f"  Warning: Directory not found.")
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
    print(f"Found {len(txt_files)} .txt files.")
    return sorted(txt_files)
