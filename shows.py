import os
from utils import (
    find_duplicates,
    get_directories_from_user,
    get_directories_from_file,
    output_report,
    interactive_deletion,
    SUPPORTED_SHOW_EXTENSIONS
)

def process_shows():
    """Handle show-specific duplicate detection logic"""
    print("\n=== TV Show Duplicate Finder ===")
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
            
    # ... rest of the function ...

if __name__ == "__main__":
    # This allows the file to be imported without running the main code
    process_shows()
