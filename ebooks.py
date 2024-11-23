import os
from utils import (
    find_duplicates,
    get_directories_from_user,
    get_directories_from_file,
    output_report,
    interactive_deletion,
    SUPPORTED_EBOOK_EXTENSIONS
)

def process_ebooks():
    """Handle ebook-specific duplicate detection logic"""
    print("\n=== Ebook Duplicate Finder ===")
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
        return

    if not directories:
        print("No directories provided. Exiting.")
        return

    print(f"\nScanning the following directories for ebook duplicates:\n{directories}")

    # Find ebook duplicates
    ebook_duplicates = {}
    all_duplicates = find_duplicates(directories, SUPPORTED_EBOOK_EXTENSIONS)
    
    # Filter for ebook files only
    for title, paths in all_duplicates.items():
        ebook_paths = [p for p in paths if any(p.lower().endswith(ext) for ext in SUPPORTED_EBOOK_EXTENSIONS)]
        if len(ebook_paths) > 1:
            ebook_duplicates[title] = ebook_paths

    if not ebook_duplicates:
        print("No ebook duplicates found.")
        return

    # User chooses mode: report only or interactive deletion
    print("\nWhat would you like to do?")
    print("1. Generate a report only")
    print("2. Interactively review and delete duplicates")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        # Generate report only
        output_file = "ebook_duplicates.txt"
        output_report(ebook_duplicates, output_file)
        print(f"\nReport saved to {output_file}.")
    elif mode == "2":
        # Interactive deletion
        print("\nStarting interactive deletion...")
        interactive_deletion(ebook_duplicates)
    else:
        print("Invalid choice. Exiting.")
