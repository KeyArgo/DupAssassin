import os
import sys
from utils import get_directories_from_user, get_directories_from_file
import movies
import shows
from audiobooks import main as process_audiobooks
import ebooks

def main_menu():
    while True:
        print("\n=== DupAssassin: Media Duplicate Finder ===")
        print("A tool to find and eliminate duplicate media files")
        print("\nSelect media type to scan:")
        print("1. Movies")
        print("2. TV Shows")
        print("3. Audiobooks")
        print("4. Ebooks")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            movies.process_movies()
        elif choice == "2":
            shows.process_shows()
        elif choice == "3":
            process_audiobooks()
        elif choice == "4":
            ebooks.process_ebooks()
        elif choice == "5":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")

        # Ask if user wants to continue
        if input("\nWould you like to assassinate more duplicates? (y/n): ").lower() != 'y':
            print("Goodbye!")
            break

def get_directories():
    """Get directories from user input or file"""
    print("\nChoose how to provide directories:")
    print("1. Enter directories interactively")
    print("2. Read directories from a file")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        return get_directories_from_user()
    elif choice == "2":
        file_path = input("Enter the path to the file containing directories: ").strip()
        return get_directories_from_file(file_path)
    else:
        print("Invalid choice. Exiting.")
        return []

if __name__ == "__main__":
    # Create banner
    print("""
╔═══════════════════════════════════════════╗
║           DupAssassin v1.0.1-α            ║
║      Your Duplicate File Eliminator       ║
╚═══════════════════════════════════════════╝
    """)
    
    # Ensure the Duplicates directory exists
    os.makedirs("Duplicates", exist_ok=True)
    
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
