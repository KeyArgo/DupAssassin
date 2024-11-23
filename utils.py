import os
import re
import shutil
from collections import defaultdict

# File extension constants
SUPPORTED_MOVIE_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov"]
SUPPORTED_SHOW_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov"]
SUPPORTED_AUDIOBOOK_EXTENSIONS = [".mp3", ".m4b", ".aac"]
SUPPORTED_EBOOK_EXTENSIONS = [".epub", ".mobi", ".pdf", ".azw", ".azw3", ".djvu"]

def normalize_name(name):
    """Normalize file or folder names by removing metadata and unnecessary characters."""
    name = re.sub(r'\(.*?\)|\[.*?\]|\d{4,}', '', name)  # Remove content in brackets or long numbers
    name = re.sub(r'(\.pdf|\.epub|\.mobi|\.azw3|\.djvu|\.mp3|\.m4b|\.mp4|\.mkv|\.avi)$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[.-]', ' ', name)  # Replace dots and dashes with spaces
    return ''.join(e for e in name if e.isalnum() or e.isspace()).lower()

def find_duplicates(directories, extensions):
    """Detect duplicates for a specific media type across multiple directories."""
    media_dict = defaultdict(list)

    for directory in directories:
        if not os.path.isdir(directory):
            print(f"Directory not found: {directory}")
            continue

        for root, dirs, files in os.walk(directory):
            # Skip @eaDir directories
            if '@eaDir' in dirs:
                dirs.remove('@eaDir')  # This prevents os.walk from traversing into @eaDir
            
            # Skip if current directory is @eaDir
            if '@eaDir' in root:
                continue

            # Check folders (excluding @eaDir)
            for folder in dirs:
                if '@eaDir' not in folder:  # Skip @eaDir folders
                    normalized_title = normalize_name(folder)
                    media_dict[normalized_title].append(os.path.join(root, folder))

            # Check individual files (excluding those in @eaDir)
            for file in files:
                if file.endswith(tuple(extensions)) and '@eaDir' not in file:
                    normalized_title = normalize_name(file)
                    media_dict[normalized_title].append(os.path.join(root, file))

    # Find duplicates
    duplicates = {title: paths for title, paths in media_dict.items() if len(paths) > 1}
    return duplicates

def output_report(duplicates, output_file):
    """Outputs duplicates to a text file for review."""
    with open(output_file, "w") as f:
        if duplicates:
            f.write("Duplicate files and folders found:\n")
            for title, paths in duplicates.items():
                f.write(f"\nTitle: {title}\n")
                for path in paths:
                    size = get_size(path)
                    f.write(f"  - {path} ({size:.2f} MB)\n")
        else:
            f.write("No duplicates found.\n")
    print(f"Report generated: {output_file}")

def get_size(path):
    """Get size of a file or directory in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    else:
        return sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(path) 
                  for f in filenames) / (1024 * 1024)

def interactive_deletion(duplicates):
    """Allow the user to interactively decide which duplicates to delete."""
    for title, paths in duplicates.items():
        print(f"\nDuplicates found for: {title}")
        for i, path in enumerate(paths):
            size = get_size(path)
            print(f"  {i + 1}: {path} ({size:.2f} MB)")

        keep = input("Enter the number of the file/folder to keep (or press Enter to skip): ").strip()
        if keep.isdigit():
            keep_index = int(keep) - 1
            if 0 <= keep_index < len(paths):
                for i, path in enumerate(paths):
                    if i != keep_index:
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                            print(f"Deleted directory: {path}")
                        elif os.path.isfile(path):
                            os.remove(path)
                            print(f"Deleted file: {path}")
            else:
                print("Invalid selection. Skipping...")
        else:
            print("Skipping this duplicate set.")

def get_directories_from_user():
    """Get a list of directories from the user interactively."""
    print("Enter the directories to scan for duplicates, separated by commas:")
    directories = input("Directories: ").strip().split(",")
    return [d.strip() for d in directories]

def get_directories_from_file(file_path):
    """Get a list of directories from an input file."""
    try:
        with open(file_path, "r") as f:
            directories = [line.strip() for line in f.readlines() if line.strip()]
        return directories
    except FileNotFoundError:
        print(f"Input file not found: {file_path}")
        return []
