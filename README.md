# DupAssassin

A Python tool for finding and managing duplicate media files, with specialized handling for audiobooks and OpenLibrary integration.

## Features

- Advanced duplicate detection with intelligent pattern matching
- OpenLibrary API integration with rate limiting
- Comprehensive reporting system:
  - HTML reports with CSS styling
  - JSON export for data persistence
  - Directory-based statistics
  - Space savings calculations
  - Processing statistics and memory monitoring

### Audiobook-Specific Features
- Smart chapter and series pattern detection
- OpenLibrary metadata enrichment
- Rate-limited API integration (100 requests/5 minutes)
- Multiple format support (.mp3, .m4b, .aac)
- Advanced pattern matching for:
  - Series detection (e.g., "Book #1", "Volume 2/3")
  - Chapter identification
  - Track numbering
  - Multiple narrators/versions

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`

## Usage

1. Run `python DupAssassin.py`
2. Select media type (Audiobooks, Movies, TV Shows, Ebooks)
3. Choose directories to scan
4. Review potential duplicates with options:
   - View duplicate groups
   - View directory statistics
   - View file patterns
   - Process with OpenLibrary verification

## Technical Features

### Performance Optimization
- Memory usage monitoring and garbage collection
- Multi-threaded processing
- Rate-limited API requests
- Progress tracking with ETA

### Error Handling
- Graceful API timeout handling
- Interrupt signal management
- Invalid file structure detection
- Logging with configurable levels

### Reporting
- Detailed HTML reports
- JSON data export
- Directory-based analysis
- Space savings calculations
- Processing statistics:
  - File processing rates
  - API success rates
  - Memory usage tracking
  - Error logging

## Requirements
- Python 3.8+
- OpenLibrary API access
- Required packages in requirements.txt

## License
MIT License - See LICENSE file