# DupAssassin

A Python tool for finding and managing duplicate media files, with specialized handling for audiobooks.

## Features

- Detects duplicate files across multiple directories
- Specialized audiobook handling with metadata support
- Support for multiple media types:
  - Movies
  - TV Shows
  - Audiobooks (with enhanced metadata)
  - Ebooks
- Interactive or report-only modes
- Synology NAS compatibility (@eaDir handling)

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`

## Usage

Run `python DupAssassin.py` and follow the interactive prompts.

## Media Type Support

### Audiobooks
- Enhanced metadata support
- OpenLibrary integration
- Intelligent duplicate detection
- Format: .mp3, .m4b, .aac

### Other Media
- Movies: .mp4, .mkv, .avi, .mov
- TV Shows: .mp4, .mkv, .avi, .mov
- Ebooks: .epub, .mobi, .pdf, .azw, .azw3, .djvu

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Requirements

- tqdm
- requests
- python-dateutil