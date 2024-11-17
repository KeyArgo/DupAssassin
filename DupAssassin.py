import os
import re
import shutil
from collections import defaultdict

SUPPORTED_MOVIE_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov"]
SUPPORTED_AUDIOBOOK_EXTENSIONS = [".mp3", ".m4b", ".aac"]
SUPPORTED_EBOOK_EXTENSIONS = [".pdf", ".epub", ".mobi", ".azw3", ".djvu"]

def normalize_name(name):
    """
    Normalize file or folder names by removing metadata and unnecessary characters.
    """
    name = re.sub(r'\(.*?\)|\[.*?\]|\d{4,}', '', name)  # Remove content in brackets or long numbers
    name = re.sub(r'(\.pdf|\.epub|\.mobi|\.azw3|\.djvu|\.mp3|\.m4b|\.mp4|\.mkv|\.avi)$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[.-]', ' ', name)  # Replace dots and dashes with spaces
    return ''.join(e for e in name if e.isalnum() or e.isspace()).lower()

def find_duplicates(directories, extensions):
    """
    Detect duplicates for a specific media type across multiple directories.
    """
    media_dict = defaultdict(list)

    for directory in directories:
        if not os.path.isdir(directory):
            print(f"Directory not found: {directory}")
            continue

        for root, dirs, files in os.walk(directory):
            # Check folders
            for folder in dirs:
                normalized_title = normalize_name(folder)
                media_dict[normalized_title].append(os.path.join(root, folder))

            # Check individual files
            for file in files:
                if file.endswith(tuple(extensions)):
                    normalized_title = normalize_name(file)
                    media_dict[normalized_title].append(os.path.join(root, file))

    # Find duplicates
    duplicates = {title: paths for title, paths in media_dict.items() if len(paths) > 1}
    return duplicates

def output_report(duplicates, output_file):
    """
    Outputs duplicates to a text file for review.
    """
    with open(output_file, "w") as f:
        if duplicates:
            f.write("Duplicate files and folders found:\n")
            for title, paths in duplicates.items():
                f.write(f"\nTitle: {title}\n")
                for path in paths:
                    f.write(f"  - {path}\n")
        else:
            f.write("No duplicates found.\n")
    print(f"Report generated: {output_file}")

def interactive_deletion(duplicates):
    """
    Allow the user to interactively decide which duplicates to delete.
    """
    for title, paths in duplicates.items():
        print(f"\nDuplicates found for: {title}")
        for i, path in enumerate(paths):
            size = os.path.getsize(path) if os.path.isfile(path) else sum(
                os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(path) for f in filenames
            )
            print(f"  {i + 1}: {path} ({size / (1024 * 1024):.2f} MB)")

        keep = input("Enter the number of the file/folder to keep (or press Enter to skip): ").strip()
        if keep.isdigit():
            keep_index = int(keep) - 1
            for i, path in enumerate(paths):
                if i != keep_index:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"Deleted directory: {path}")
                    elif os.path.isfile(path):
                        os.remove(path)
                        print(f"Deleted file: {path}")
        else:
            print("No files deleted for this duplicate set.")

def get_directories_from_user():
    """
    Get a list of directories from the user interactively.
    """
    print("Enter the directories to scan for duplicates, separated by commas:")
    directories = input("Directories: ").strip().split(",")
    return [d.strip() for d in directories]

def get_directories_from_file(file_path):
    """
    Get a list of directories from an input file.
    """
    try:
        with open(file_path, "r") as f:
            directories = [line.strip() for line in f.readlines() if line.strip()]
        return directories
    except FileNotFoundError:
        print(f"Input file not found: {file_path}")
        return []

if __name__ == "__main__":
    print("Choose how to provide directories:")
    print("1. Enter directories interactively")
    print("2. Read directories from a file")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        directories = get_directories_from_user()
    elif choice == "2":
        file_path = input("Enter the path to the file containing directories: ").strip()
        directories = get_directories_from_file(file_path)
    else:
        print("Invalid choice. Exiting.")
        exit()

    if not directories:
        print("No directories provided. Exiting.")
        exit()

    print(f"\nScanning the following directories:\n{directories}")

    # Find duplicates for each media type
    output_file = "duplicates_report.txt"
    all_duplicates = {}

    print("\nScanning for movie duplicates...")
    all_duplicates['Movies'] = find_duplicates(directories, SUPPORTED_MOVIE_EXTENSIONS)

    print("Scanning for audiobook duplicates...")
    all_duplicates['Audiobooks'] = find_duplicates(directories, SUPPORTED_AUDIOBOOK_EXTENSIONS)

    print("Scanning for ebook duplicates...")
    all_duplicates['Ebooks'] = find_duplicates(directories, SUPPORTED_EBOOK_EXTENSIONS)

    # User chooses mode: report only or interactive deletion
    print("\nWhat would you like to do?")
    print("1. Generate a report only")
    print("2. Interactively review and delete duplicates")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        # Generate report only
        output_report(all_duplicates['Movies'], output_file)
        output_report(all_duplicates['Audiobooks'], output_file)
        output_report(all_duplicates['Ebooks'], output_file)
        print(f"\nReports saved to {output_file}.")
    elif mode == "2":
        # Interactive deletion
        print("\nStarting interactive deletion...")
        interactive_deletion(all_duplicates['Movies'])
        interactive_deletion(all_duplicates['Audiobooks'])
        interactive_deletion(all_duplicates['Ebooks'])
    else:
        print("Invalid choice. Exiting.")
