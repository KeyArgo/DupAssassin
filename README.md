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
- Enhanced reporting capabilities:
  - HTML reports with visual formatting
  - JSON export of potential duplicates
  - Directory-based statistics
  - Space savings calculations

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
- Advanced pattern matching for:
  - Series detection
  - Chapter identification
  - Book numbering
  - Multiple formats

### Other Media
- Movies: .mp4, .mkv, .avi, .mov
- TV Shows: .mp4, .mkv, .avi, .mov
- Ebooks: .epub, .mobi, .pdf, .azw, .azw3, .djvu

## Reports
- HTML reports with CSS styling
- JSON export for data persistence
- Directory-based statistics
- Space savings calculations
- Processing statistics:
  - File processing rates
  - API success rates
  - Error tracking
  - Memory usage

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Requirements
- Python 3.8+
- OpenLibrary API access
- Required Python packages in requirements.txt