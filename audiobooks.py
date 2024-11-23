import os
from collections import defaultdict
from pathlib import Path
import psutil
from tqdm import tqdm
import re
import requests
from time import sleep
from datetime import datetime
import time
import multiprocessing as mp
from functools import partial
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Optional, TypedDict, Sequence, Any, Tuple, Union, Mapping, Protocol, cast, TypeVar, overload
from urllib.parse import quote
from functools import lru_cache
import gc
import signal
import sys
import threading
from concurrent.futures import wait
from library_metadata import LibraryMetadataProcessor, BookMetadata
import json

SUPPORTED_EXTENSIONS = ['.mp3', '.m4b', '.aac']

# Compile patterns once at module level
COMPILED_CHAPTER_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in [
    # Base patterns
    r'^chapter\s*\d+[a-z]?(?:\s*[-_]?\s*\w+)*$',    
    r'^part\s*\d+(?:\.mp3)?$',                       
    r'^\d{1,3}$',                                    
    r'^\d+\s*(?:track|cd)\s+\d+$',                   
    r'^(?:intro|outro|prologue|epilogue|introduction|acknowledgements)$',
    r'^\d+\s*[-_]?\s*(?:intro|outro|prologue|epilogue|chapter\s+\d+)',
    r'^\d+\s+\d+\s+track\s+\d+$',
    r'^\d+_\s*book\s+[ivx]+\s*-\s*chapter',
    r'^s\d+\s*-\s*[^-]+\s*-\s*\d+',
    r'^chapter\s+[a-z\s-]+$',
    r'^\d+[a-z]?\s*-\s*\d+$',
    r'.*?book\s+\d+.*?chapter\s+\d+',
    r'^\d+[-_]\d+$',
    r'.*?part\s+\d+.*?chapter\s+\d+',
    r'.*?part\s+\d+.*?chapter\s+\d+',
    # Track patterns
    r'^\d+-track\s*\d+',                    # 02-Track 2
    r'^\d+\s*track\s*\d+'
]]

COMPILED_SERIES_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in [
    r'^(?:book|saga|series)\s*(?:#|\d+)',  
    r'^#\d+\s*-',                          # #34 - Title
    r'\d+[A-Z]?\s*-',                      # 13A - Title
    r'S\d+\s*-',                           # S34 - Title
    r'.*?(?:book|saga|series)\s*\d+.*?-',   # Vorkosigan Saga Book 1 - 
    r'.*?(?:#\d+|book\s+\d+).*?$',          # Origins #1, Book 4
    r'^\d+[a-z]?\s*-',                      # 01 -, 13A -
    r'.*?(?:part|volume)\s*\d+\s*(?:of|\/)\s*\d+', # Part 1 of 2, Volume 1/3
    r'^\d+\s*[a-z].*?\[.*?\]'              # 1Girl with Dragon Tattoo[HTD 2017]
]]

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path('Logs')  # Changed from 'logs' to 'Logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'duplicate_finder_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,  # Change from DEBUG to INFO
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class SeriesInfo(TypedDict):
    series: Optional[str]
    book_num: Optional[str]

class BookInfo(TypedDict):
    title: str
    author: str
    path: str
    filename: str
    series_info: SeriesInfo

class FileInfoProtocol(Protocol):
    name: str
    path: str
    size: int
    original_name: Optional[str]
    book_info: Optional[Any]

class FileInfo(TypedDict):
    name: str
    path: str
    size: int
    original_name: Optional[str]
    book_info: Optional[Any]

class DuplicateGroup(TypedDict):
    files: List[FileInfo]
    metadata: BookMetadata
    suggested_name: str
    openlibrary_url: Optional[str]

# Type definitions
GroupType = Union[List[FileInfo], DuplicateGroup]
InitialGroupType = Dict[str, List[Dict[str, Any]]]  # Fixed: Added missing closing bracket

# The @lru_cache decorator is valid Python syntax - ignore the type checker error
@lru_cache(maxsize=1000)
def get_normalized_path(filepath: str) -> Tuple[str, str, str]:
    """Cache frequently accessed path components"""
    filename = os.path.basename(filepath)
    parent_dir = os.path.basename(os.path.dirname(filepath))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
    return filename, parent_dir, grandparent

@lru_cache(maxsize=10000)
def normalize_filename(filepath: str, total_files: Optional[int] = None) -> str:
    """Normalize by audiobook directory rather than individual files"""
    try:
        # Get the parent directory as the book directory
        book_dir = os.path.dirname(filepath)
        book_name = os.path.basename(book_dir)
        
        # If the directory is numbered (like "01 - Book Name"), remove the number
        book_name = re.sub(r'^\d+\s*[-_.]?\s*', '', book_name)
        
        # Remove common audiobook markers
        book_name = re.sub(r'\((?:read by|narrated by|unabridged|abridged).*?\)', '', book_name, flags=re.IGNORECASE)
        
        # Clean up remaining punctuation and whitespace
        book_name = re.sub(r'[-_\s]+', ' ', book_name).strip()
        
        return book_name.lower()
        
    except Exception as e:
        logger.error(f"Error normalizing {filepath}: {e}")
        return filepath

def process_files(files: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    file_groups = defaultdict(list)
    start_time = time.time()
    processed_count = 0
    last_update = time.time()
    update_interval = 10.0  # Update every 10 seconds
    
    # Process files in batches
    with tqdm(total=len(files), desc="Processing files") as pbar:
        for filepath in files:
            try:
                size = os.path.getsize(filepath)
                normalized = normalize_filename(filepath)
                filename = os.path.basename(filepath)
                
                file_info = {
                    'path': filepath,
                    'name': filename,
                    'size': size,
                    'normalized_name': normalized
                }
                
                file_groups[normalized].append(file_info)
                processed_count += 1
                pbar.update(1)
                
                # Update progress stats periodically
                current_time = time.time()
                if current_time - last_update >= update_interval:
                    elapsed_time = current_time - start_time
                    if elapsed_time > 0:  # Prevent division by zero
                        rate = processed_count / elapsed_time
                        logger.info(f"Progress: {processed_count} files ({rate:.1f} files/sec) - Phase: scanning")
                        last_update = current_time
                
            except OSError as e:
                logger.warning(f"Error processing {filepath}: {e}")
    
    return {name: files for name, files in file_groups.items() if len(files) > 1}

def get_directory_size(path):
    """Get total size of directory or file"""
    if os.path.isfile(path):
        return os.path.getsize(path)
    
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                continue
    return total

def get_directories():
    """Interactively get directories to scan"""
    directories = []
    print("\nEnter directories to scan (one per line, empty line to finish):")
    while True:
        directory = input("> ").strip()
        if not directory:
            break
            
        if os.path.isdir(directory):
            directories.append(directory)
        else:
            print(f"Warning: '{directory}' is not a valid directory")
    
    return directories

def find_duplicates(directories: List[str], extensions: List[str] = SUPPORTED_EXTENSIONS) -> Dict[str, List[Dict[str, Any]]]:
    """Find duplicate audio files across directories."""
    logger.info("\n" + "="*50)
    logger.info("Starting duplicate search")
    logger.info("="*50)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Directories: {directories}")
    logger.info(f"  Extensions: {extensions}")
    logger.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*50 + "\n")
    
    start_time = time.time()
    all_files: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_files = 0
    skipped_files = 0
    
    for directory in directories:
        logger.info(f"\nScanning directory: {directory}")
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    filepath = os.path.join(root, file)
                    try:
                        file_info = {
                            'path': filepath,
                            'name': file,
                            'size': os.path.getsize(filepath)
                        }
                        normalized_name = normalize_filename(filepath)
                        logger.debug(f"Normalized '{file}' -> '{normalized_name}'")
                        all_files[normalized_name].append(file_info)
                        total_files += 1
                    except Exception as e:
                        logger.error(f"Error processing file {filepath}: {e}")
                        skipped_files += 1
                        continue
                    
                    if total_files % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = total_files / elapsed
                        logger.info(f"Progress: {total_files} files ({rate:.1f} files/sec)")

    # Log final statistics
    scan_time = time.time() - start_time
    logger.info("\nScan Summary:")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Skipped files: {skipped_files}")
    logger.info(f"  Time taken: {scan_time:.2f} seconds")
    logger.info(f"  Processing rate: {total_files/scan_time:.1f} files/sec")
    
    # Log duplicate findings
    duplicates = {k: v for k, v in all_files.items() if len(v) > 1}
    logger.info(f"\nFound {len(duplicates)} potential duplicate groups")
    
    return duplicates

def group_files(files: Sequence[Dict]) -> Dict[str, List[FileInfo]]:
    """Group files with improved normalization"""
    groups: Dict[str, List[FileInfo]] = defaultdict(list)
    
    for file_data in files:
        try:
            book_info = extract_book_info(
                file_data.get('name', os.path.basename(file_data['path'])),
                file_data['path']
            )
            
            file_info: FileInfo = {
                'name': file_data.get('name', ''),
                'path': file_data['path'],
                'size': file_data['size'],
                'original_name': file_data.get('original_name'),
                'book_info': book_info
            }
            
            # Build normalized key
            key_parts = []
            if book_info['author']:
                key_parts.append(book_info['author'])
            if book_info['title']:
                key_parts.append(book_info['title'])
            if book_info['series_info']['series']:
                key_parts.append(f"{book_info['series_info']['series']}")
                if book_info['series_info']['book_num']:
                    key_parts.append(f"book {book_info['series_info']['book_num']}")
            
            normalized_key = normalize_filename(' - '.join(key_parts) if key_parts else file_info['name'])
            groups[normalized_key].append(file_info)
            
        except Exception as e:
            logger.error(f"Error grouping file {file_data.get('path', 'Unknown')}: {e}")
            continue
    
    return groups

def extract_book_info(filename: str, filepath: str) -> BookInfo:
    """Extract book information from filename with improved series and author handling"""
    name = os.path.splitext(filename)[0].lower()
    
    # Initialize series info
    series_info: SeriesInfo = {
        'series': None,
        'book_num': None
    }
    
    # Extract series info if present
    series_patterns = [
        r'(?i)book\s*(\d+)',
        r'(?i)volume\s*(\d+)',
        r'(?i)[,\s](\d+)(?:st|nd|rd|th)?\s*(?:book|vol)',
        r'(?i)series\s*(?:book)?\s*(\d+)',
    ]
    
    for pattern in series_patterns:
        if match := re.search(pattern, name):
            series_info['book_num'] = match.group(1)
            # Extract series name if possible
            series_name = re.search(r'(.*?)\s*(?:book|volume|series)', name, re.IGNORECASE)
            if series_name:
                series_info['series'] = series_name.group(1).strip()
            name = re.sub(pattern, '', name)
    
    # Rest of the function remains the same until return
    
    return {
        'title': name.strip(),
        'author': '',
        'path': filepath,
        'filename': filename,
        'series_info': series_info
    }

def safe_api_call(url: str, timeout: int = 10, max_retries: int = 3, initial_delay: float = 1.0) -> Optional[Dict]:
    """Make API call with exponential backoff retry logic"""
    for attempt in range(max_retries):
        delay = initial_delay * (2 ** attempt)  # Exponential backoff
        
        if attempt > 0:
            logger.debug(f"Retry attempt {attempt + 1} after {delay:.1f}s delay")
        
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
            
        except requests.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1} for URL: {url}")
            
        except requests.RequestException as e:
            logger.warning(f"Request failed on attempt {attempt + 1}: {str(e)}")
            
            # Only check status code if response exists
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 429:  # Too Many Requests
                    logger.warning("Rate limit hit, increasing delay")
                    delay *= 2
                elif e.response.status_code >= 500:  # Server errors
                    logger.warning("Server error, will retry")
                else:
                    logger.error(f"Client error {e.response.status_code}, skipping retries")
                    return None
            else:
                logger.error("No response object available")
                return None
                    
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None
            
        sleep(delay)
    
    logger.error(f"Failed after {max_retries} attempts: {url}")
    return None

def search_ol_api(query: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Search OpenLibrary API with the given query parameters"""
    try:
        # Build search query from parameters
        search_terms = []
        if query.get('title'):
            search_terms.append(query['title'])
        if query.get('author'):
            search_terms.append(query['author'])
        if query.get('series'):
            search_terms.append(query['series'])
            
        search_query = ' '.join(search_terms)
        
        # Skip if query is too short or contains only numbers
        if len(search_query) < 3 or search_query.replace(' ', '').isdigit():
            logger.debug(f"Skipping invalid query: {search_query}")
            return None
            
        # URL encode the search query
        encoded_query = quote(search_query)
        search_url = f"https://openlibrary.org/search.json?q={encoded_query}&fields=key,title,author_name,first_publish_date,isbn,edition_key,number_of_pages_median&limit=1"
        
        # Use existing safe_api_call function
        response_data = safe_api_call(search_url, timeout=5)
        
        if not response_data or 'docs' not in response_data or not response_data['docs']:
            logger.debug(f"No results found for: {search_query}")
            return None
            
        book_data = response_data['docs'][0]
        
        # Format the response
        return {
            'title': book_data.get('title'),
            'author': book_data.get('author_name', ['Unknown'])[0],
            'year': book_data.get('first_publish_date', '').split(',')[0].strip(),
            'editions': len(book_data.get('edition_key', [])),
            'isbn': book_data.get('isbn', [None])[0],
            'work_id': book_data.get('key', '').split('/')[-1]
        }
        
    except Exception as e:
        logger.error(f"Error in OpenLibrary API search: {e}")
        return None

def clean_search_term(term: str) -> str:
    """Clean up search terms for better matching"""
    # Remove common audiobook indicators
    term = re.sub(r'\b(mp3|m4b|audiobook|unabridged|chapterized)\b', '', term, flags=re.IGNORECASE)
    # Remove parenthetical info like (Book 1) or (Series Name #1)
    term = re.sub(r'\([^)]*\)', '', term)
    # Remove file info like 32k, 128k, etc.
    term = re.sub(r'\b\d+k\b', '', term)
    # Remove timestamps and common separators
    term = re.sub(r'\d{2}\.\d{2}\.\d{2}', '', term)
    term = re.sub(r'[-_]', ' ', term)
    return term.strip()

class OpenLibraryClient:
    def __init__(self):
        self.session = requests.Session()
        self.rate_limit = 100  # requests
        self.rate_window = 300  # 5 minutes in seconds
        self.request_timestamps: List[float] = []
        self.min_delay = 3.1  # Slightly over 3 seconds to be safe
        self.timeout = 5
        
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed 100 requests per 5 minutes"""
        current_time = time.time()
        
        # Remove timestamps older than our window
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < self.rate_window
        ]
        
        # If we're at the limit, wait until oldest request expires
        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = self.rate_window - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
        
        # Always wait minimum delay between requests
        last_request = self.request_timestamps[-1] if self.request_timestamps else 0
        time_since_last = current_time - last_request
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        
        # Add current request timestamp
        self.request_timestamps.append(time.time())
    
    def search(self, query: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Search OpenLibrary with strict rate limiting"""
        url = "https://openlibrary.org/search.json"
        
        try:
            self._wait_for_rate_limit()
            
            response = self.session.get(
                url,
                params={
                    **query,
                    'fields': 'key,title,author_name,first_publish_date,isbn',
                    'limit': 1
                },
                timeout=self.timeout
            )
            
            if response.status_code == 429:  # Too Many Requests
                logger.warning("Rate limit exceeded, waiting for reset...")
                time.sleep(self.min_delay * 2)  # Wait extra time on rate limit
                return None
                
            response.raise_for_status()
            return response.json()
            
        except requests.Timeout:
            logger.warning(f"Timeout for query: {query}")
            return None
            
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

# Initialize the global client
ol_client = OpenLibraryClient()

def search_openlibrary(book_info: BookInfo) -> Optional[Dict[str, Any]]:
    """Search OpenLibrary using the global client"""
    if not book_info:
        return None
        
    # Clean and prepare search terms
    title = clean_search_term(book_info.get('title', ''))
    author = clean_search_term(book_info.get('author', ''))
    
    # Try different search strategies
    queries = [
        {'title': title, 'author': author} if title and author else None,
        {'q': f"{author} {title}"} if author and title else None,
        {'title': title} if title else None,
        {'author': author} if author else None
    ]
    
    for query in queries:
        if not query:
            continue
            
        result = ol_client.search(query)
        if result and result.get('docs'):
            return result
            
    return None

def group_by_directory(files):
    """Group files by their parent directory"""
    groups = defaultdict(list)
    for f in files:
        parent = str(Path(f['path']).parent)
        groups[parent].append(f)
    return groups

def print_duplicates(duplicates, include_ol_info=False, output_file=None) -> str:
    """Generate duplicate files report in a readable format and return as string"""
    output = []
    if not duplicates:
        output.append("\nNo duplicates found.")
    else:
        output.append("\nDuplicate Groups Found:")
        output.append("=" * 80)
        total_wasted_space = 0
        
        for name, group in sorted(duplicates.items()):
            files = group.get('files', group) if include_ol_info else group
            
            if include_ol_info and isinstance(group, dict) and 'metadata' in group:
                ol_info = group['metadata']
                output.append(f"\nBook: {name}")
                output.append("-" * 80)
                output.append(f"Title: {ol_info.get('title', 'Unknown')}")
                output.append(f"Author: {ol_info.get('author_name', ['Unknown'])[0]}")
                if ol_info.get('year'):
                    output.append(f"First Published: {ol_info['year']}")
                output.append(f"Known Editions: {ol_info.get('editions', 0)}")
                output.append("")
            
            total_size = sum(get_directory_size(Path(f['path']).parent) for f in files)
            wasted_space = total_size - min(get_directory_size(Path(f['path']).parent) for f in files)
            total_wasted_space += wasted_space
            
            output.append(f"Total Size: {total_size / (1024**3):.2f} GB")
            output.append(f"Potential Space Savings: {wasted_space / (1024**3):.2f} GB\n")
            
            for dir_path, dir_files in group_by_directory(files).items():
                output.append(f"  Directory: {dir_path}")
                dir_size = get_directory_size(dir_path)
                output.append(f"  Size: {dir_size / (1024**2):.2f} MB")
                for f in dir_files:
                    if 'name' not in f:
                        logger.warning("Missing 'name' key in:", f)
                        continue
                    output.append(f"    - {f['name']}")
            output.append("\n" + "=" * 80)
            
        output.append("\n" + "=" * 80)
        output.append(f"Total Potential Space Savings: {total_wasted_space / (1024**3):.2f} GB")

    # Convert output list to string
    report_text = '\n'.join(output)
    
    # Write to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    return report_text

def process_file(file_tuple, extensions):
    """Simplified file processing focusing only on path and name"""
    root, file = file_tuple
    if not file.endswith(tuple(extensions)):
        return None
        
    try:
        filepath = os.path.join(root, file)
        book_info = {
            'title': os.path.splitext(file)[0],
            'path': filepath,
            'original_name': file
        }
        return book_info
    except Exception:
        return None
    
class DuplicateFinder:
    def __init__(self):
        self.stats = {
            'files_processed': 0,
            'start_time': time.time(),
            'last_progress_time': time.time(),
            'scan_phase': 'initializing',
            'api_calls': 0,
            'duplicates_found': 0,
            'errors': defaultdict(int),
            'potential_duplicates': 0
        }
        self.chunk_size = 1000
        self.timeout = 120
        self.interrupted = False

    def _show_scan_progress(self) -> None:
        """Show current scan progress with rate limiting"""
        current_time = time.time()
        
        # Only update progress every 5 seconds
        if current_time - self.stats['last_progress_time'] < 5:
            return
        
        elapsed = current_time - self.stats['start_time']
        if elapsed > 0:
            files_per_sec = self.stats['files_processed'] / elapsed
            logger.info(f"Progress: {self.stats['files_processed']} files ({files_per_sec:.1f} files/sec) - Phase: {self.stats['scan_phase']}")
            self.stats['last_progress_time'] = current_time

    def check_interrupt(self) -> bool:
        """Check if processing should be interrupted"""
        return self.interrupted or (
            hasattr(signal_handler, 'interrupted') and 
            signal_handler.interrupted
        )

    def _analyze_groups(self, groups: Dict[str, List[FileInfo]]) -> Dict[str, List[FileInfo]]:
        """Analyze groups for actual duplicates"""
        duplicates: Dict[str, List[FileInfo]] = {}
        total_groups = len(groups)
        
        with tqdm(total=total_groups, desc="Analyzing groups", unit="groups") as pbar:
            for name, files in groups.items():
                if len(files) > 1:  # Only keep groups with multiple files
                    # Group by parent directory to avoid chapter-level duplicates
                    dir_groups: Dict[str, List[FileInfo]] = defaultdict(list)
                    for file in files:
                        parent_dir = os.path.dirname(file['path'])
                        dir_groups[parent_dir].append(file)
                    
                    # Only keep as duplicate if files exist in multiple directories
                    if len(dir_groups) > 1:
                        duplicates[name] = files
                        self.stats['potential_duplicates'] += len(dir_groups)
                
                pbar.update(1)
                
                if len(duplicates) % self.chunk_size == 0:
                    self._show_scan_progress()
        
        return duplicates

    def _normalize_and_group(self, files: List[FileInfo]) -> Dict[str, List[FileInfo]]:
        groups: Dict[str, List[FileInfo]] = defaultdict(list)
        batch_size = 100
        
        try:
            with tqdm(total=len(files), desc="Normalizing files", unit="files") as pbar:
                for i in range(0, len(files), batch_size):
                    batch = files[i:i + batch_size]
                    for file in batch:
                        normalized = normalize_filename(file['path'])
                        # Cast the dictionary to FileInfo to satisfy type checker
                        file_info = cast(FileInfo, {
                            'name': file['name'],
                            'path': file['path'],
                            'size': file['size'],
                            'original_name': file.get('original_name'),
                            'book_info': file.get('book_info')
                        })
                        groups[normalized].append(file_info)
                    pbar.update(len(batch))
        except Exception as e:
            logger.error(f"Error in normalization: {e}")
        
        # Convert defaultdict to regular dict
        return dict(groups)

    def process_directories(self, directories: List[str], extensions: List[str]) -> Dict[str, DuplicateGroup]:
        try:
            # Initial scan
            logger.info("Starting initial file scan...")
            initial_groups: InitialGroupType = find_duplicates(directories, extensions)
            
            # Convert to proper types
            typed_groups = convert_groups(initial_groups)
            
            # Save potential duplicates before OpenLibrary processing
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_potential_duplicates(typed_groups, timestamp)
            
            # Preview and confirm
            if not preview_duplicates(typed_groups, self):
                logger.info("User cancelled operation after preview")
                return {}
            
            # Process with OpenLibrary
            logger.info("Starting OpenLibrary API processing...")
            enriched_groups = process_with_openlibrary(initial_groups)
            
            # Generate final report
            self.generate_report(enriched_groups, timestamp)
            
            return enriched_groups
            
        except Exception as e:
            logger.error(f"Error in process_directories: {e}")
            return {}

    def _scan_directories(self, directories: List[str], extensions: List[str]) -> List[FileInfo]:
        """Scan directories for audio files"""
        all_files = []
        total_files = 0
        
        # First count total files for progress bar
        for directory in directories:
            for root, _, files in os.walk(directory):
                total_files += len([f for f in files if f.endswith(tuple(extensions))])
        
        with tqdm(total=total_files, desc="Scanning files", unit="files") as pbar:
            for directory in directories:
                try:
                    for root, _, files in os.walk(directory):
                        if self.check_interrupt():
                            return all_files
                            
                        matching_files = []
                        for f in files:
                            if f.endswith(tuple(extensions)):
                                try:
                                    filepath = os.path.join(root, f)
                                    file_info: FileInfo = {
                                        'name': f,
                                        'path': filepath,
                                        'size': os.path.getsize(filepath),
                                        'original_name': f,
                                        'book_info': None
                                    }
                                    matching_files.append(file_info)
                                except OSError as e:
                                    self.stats['errors']['file_access'] += 1
                                    logger.error(f"Error accessing file {f}: {e}")
                                    continue
                                    
                        all_files.extend(matching_files)
                        pbar.update(len(matching_files))
                        
                        if len(all_files) % self.chunk_size == 0:
                            self._show_scan_progress()
                            
                except Exception as e:
                    self.stats['errors']['directory_access'] += 1
                    logger.error(f"Error scanning directory {directory}: {e}")
                    continue
                    
        return all_files

    def _monitor_memory(self) -> None:
        """Monitor memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        logger.debug(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB ({memory_percent:.1f}%)")
        
        if memory_percent > 80:  # Warning threshold
            logger.warning("High memory usage detected")
            gc.collect()

    def generate_report(self, groups: Mapping[str, GroupType], timestamp: str) -> Tuple[str, str]:
        """Generate both text and HTML reports of duplicate groups"""
        # Generate text report
        report_dir = "Reports"
        os.makedirs(report_dir, exist_ok=True)
        text_file = os.path.join(report_dir, f"duplicates_{timestamp}.txt")
        
        # Generate text report and save to file
        text_report = print_duplicates(groups, include_ol_info=True, output_file=text_file)
        
        # Generate HTML report
        html_report = generate_html_report(groups, timestamp)
        
        return text_report, html_report

def format_size(size_in_bytes: float) -> str:
    """Format file size in human readable format"""
    size = float(size_in_bytes)  # Convert to float at the start
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

def output_report(duplicates: Dict[str, List[Dict[str, Any]]]) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"reports/duplicates_report_{timestamp}.txt"
    Path('reports').mkdir(exist_ok=True)
    
    total_savings = 0
    
    with open(report_file, 'w', encoding='utf-8') as f:
        for book_name, files in duplicates.items():
            try:
                # Ensure files is a list of dictionaries
                if not files or not isinstance(files[0], dict):
                    logging.warning(f"Invalid files structure for {book_name}")
                    continue
                
                # Group files by directory
                dir_groups = defaultdict(list)
                for file_info in files:
                    if isinstance(file_info, dict) and 'path' in file_info:
                        dir_path = str(Path(file_info['path']).parent)
                        dir_groups[dir_path].append(file_info)
                
                # Calculate group size and savings
                group_size = sum(file_info.get('size', 0) for file_info in files if isinstance(file_info, dict))
                potential_savings = group_size - min(
                    sum(f.get('size', 0) for f in group if isinstance(f, dict))
                    for group in dir_groups.values()
                )
                
                # Write group information
                f.write(f"\nBook: {book_name}\n")
                f.write("-" * 80 + "\n")
                
                total_savings += potential_savings
                
                # Write directory information
                for dir_path, dir_files in dir_groups.items():
                    dir_size = sum(f.get('size', 0) for f in dir_files if isinstance(f, dict))
                    f.write(f"\n  Directory: {dir_path}\n")
                    f.write(f"  Size: {format_size(dir_size)}\n")
                    
                    for file_info in dir_files:
                        if isinstance(file_info, dict) and 'path' in file_info:
                            f.write(f"    - {os.path.basename(file_info['path'])}\n")
                
            except Exception as e:
                logging.error(f"Error processing group {book_name}: {e}")
                continue
        
        # Write summary
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Total Potential Space Savings: {format_size(total_savings)}\n")

def analyze_naming_patterns(directories: List[str]) -> None:
    """Analyze and log file/folder naming patterns across the library."""
    patterns = {
        'folders': defaultdict(int),
        'files': defaultdict(int),
        'extensions': defaultdict(int),
        'series_patterns': defaultdict(int),
        'chapter_patterns': defaultdict(int)
    }
    
    logger.info("\nAnalyzing naming patterns...")
    
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            # Analyze folder patterns
            for d in dirs:
                # Log different folder name patterns
                patterns['folders'][extract_pattern(d)] += 1
                
            # Analyze file patterns
            for f in files:
                if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    # Log extension
                    ext = os.path.splitext(f)[1].lower()
                    patterns['extensions'][ext] += 1
                    
                    # Log file naming pattern
                    patterns['files'][extract_pattern(f)] += 1
                    
                    # Check for series indicators
                    if series_match := re.search(r'(?:book|saga|series).*?(\d+|one|two|three)', f.lower()):
                        patterns['series_patterns'][series_match.group(0)] += 1
                    
                    # Check for chapter indicators
                    if chapter_match := re.search(r'(?:chapter|part|disc).*?(\d+)', f.lower()):
                        patterns['chapter_patterns'][chapter_match.group(0)] += 1

    # Output analysis
    logger.info("\nNaming Pattern Analysis:")
    logger.info("=" * 50)
    
    for category, counts in patterns.items():
        logger.info(f"\n{category.upper()}:")
        for pattern, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            logger.info(f"  {pattern}: {count}")
            
def extract_pattern(name: str) -> str:
    """Extract the general pattern from a name."""
    # Replace numbers with #
    pattern = re.sub(r'\d+', '#', name)
    # Replace words with W
    pattern = re.sub(r'[a-zA-Z]+', 'W', pattern)
    return pattern

def scan_directories(directories: List[str], extensions: List[str] = SUPPORTED_EXTENSIONS) -> Tuple[List[FileInfo], int]:
    all_files: List[FileInfo] = []
    total_files = 0
    
    for directory in directories:
        for root, _, files in os.walk(directory):
            matching_files = [
                FileInfo(
                    name=f,
                    path=os.path.join(root, f),
                    size=os.path.getsize(os.path.join(root, f)),
                    original_name=None,
                    book_info=None
                )
                for f in files if f.endswith(tuple(extensions))
            ]
            all_files.extend(matching_files)
            total_files += len(matching_files)
    
    return all_files, total_files

# Initialize the processor
metadata_processor = LibraryMetadataProcessor()

def process_with_openlibrary(duplicates: InitialGroupType) -> Dict[str, DuplicateGroup]:
    results: Dict[str, DuplicateGroup] = {}
    total_groups = len(duplicates)
    successful_matches = 0
    attempted_matches = 0
    
    with tqdm(total=total_groups, desc="Processing with OpenLibrary API", unit="groups") as pbar:
        for group_name, files in duplicates.items():
            if not files:
                pbar.update(1)
                continue
                
            try:
                attempted_matches += 1
                file_info = ensure_fileinfo(files[0])
                filename = file_info['name']
                filepath = file_info['path']
                
                book_info = extract_book_info(filename, filepath)
                if not book_info:
                    logger.warning(f"Could not extract book info for {filepath}")
                    pbar.update(1)
                    continue
                    
                ol_data = search_openlibrary(book_info)
                if not ol_data:
                    logger.debug(f"No OpenLibrary match found for: {group_name}")
                    pbar.update(1)
                    continue
                
                # Get OpenLibrary URL
                ol_url = f"https://openlibrary.org{ol_data.get('key', '')}" if ol_data.get('key') else None
                if ol_url:
                    logger.info(f"Found OpenLibrary match for {group_name}: {ol_url}")
                else:
                    logger.debug(f"Found OpenLibrary data but no URL for: {group_name}")
                    
                metadata = metadata_processor.normalize_book_data(ol_data, filepath=filepath)
                if not metadata:
                    logger.warning(f"Could not normalize metadata for {filepath}")
                    pbar.update(1)
                    continue
                    
                suggested_name = metadata_processor.generate_filename(metadata)
                if not suggested_name:
                    logger.warning(f"Could not generate filename for {filepath}")
                    pbar.update(1)
                    continue
                    
                # Only count as success if we got all the way through
                successful_matches += 1
                results[group_name] = DuplicateGroup(
                    files=[ensure_fileinfo(f) for f in files],
                    metadata=metadata,
                    suggested_name=suggested_name,
                    openlibrary_url=ol_url  # Add URL to the group data
                )
                
            except Exception as e:
                logger.error(f"Error processing group {group_name}: {e}")
                
            finally:
                pbar.update(1)
    
    # Calculate and log success rate
    success_rate = (successful_matches / attempted_matches * 100) if attempted_matches > 0 else 0
    logger.info(f"OpenLibrary matching complete:")
    logger.info(f"  Total groups attempted: {attempted_matches}")
    logger.info(f"  Successful matches: {successful_matches}")
    logger.info(f"  Success rate: {success_rate:.1f}%")
    
    return results

def preview_duplicates(duplicates: Mapping[str, GroupType], finder: DuplicateFinder) -> bool:
    """Preview duplicate groups before processing with OpenLibrary"""
    print(f"\nFound {len(duplicates)} potential duplicate groups")
    
    while True:
        print("\nOptions:")
        print("1. View potential duplicates")
        print("2. View directory statistics")
        print("3. View file patterns")
        print("4. Continue with OpenLibrary verification")
        print("5. Exit")
        print("6. Show processing status")
        
        choice = input("\nEnter choice (1-6): ")
        
        if choice == '1':
            _show_duplicate_groups(duplicates)
        elif choice == '2':
            _show_directory_stats(duplicates)
        elif choice == '3':
            _show_file_patterns(duplicates)
        elif choice == '4':
            return True
        elif choice == '5':
            sys.exit(0)  # Exit immediately without further prompts
        elif choice == '6':
            print("\nProcessing Status:")
            print(f"Files processed: {finder.stats['files_processed']}")
            print(f"Processing rate: {finder.stats['files_processed'] / (time.time() - finder.stats['start_time']):.1f} files/sec")
            print(f"Time since last progress: {time.time() - finder.stats['last_progress_time']:.1f} seconds")
        else:
            print("Invalid choice. Please try again.")

def preview_initial_scan(finder: DuplicateFinder, groups: Mapping[str, GroupType]) -> bool:
    """Preview initial scan results before OpenLibrary processing"""
    while True:
        print("\nInitial Scan Results:")
        print(f"Found {len(groups)} potential duplicate groups")
        print("\nOptions:")
        print("1. View potential duplicate groups")
        print("2. View directory statistics")
        print("3. Continue processing")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ")
        
        if choice == '1':
            for name, group in groups.items():
                print(f"\nGroup: {name}")
                files = group['files'] if isinstance(group, dict) else group
                for file in files:
                    print(f"  {file['path']}")
        elif choice == '2':
            total_size = sum(
                sum(f['size'] for f in (group['files'] if isinstance(group, dict) else group))
                for group in groups.values()
            )
            print(f"\nTotal groups: {len(groups)}")
            print(f"Total size: {format_size(total_size)}")
        elif choice == '3':
            return True
        elif choice == '4':
            return False

def _show_directory_stats(groups: Mapping[str, GroupType]) -> None:
    dir_stats: Dict[str, Dict[str, Union[int, float]]] = defaultdict(lambda: {'files': 0, 'size': 0})
    
    for group_data in groups.values():
        files = group_data['files'] if isinstance(group_data, dict) else group_data
        for file in files:
            dir_path = str(Path(file['path']).parent)
            dir_stats[dir_path]['files'] = int(dir_stats[dir_path]['files']) + 1
            dir_stats[dir_path]['size'] = int(dir_stats[dir_path]['size']) + int(file['size'])
    
    print("\nDirectory Statistics:")
    for dir_path, stats in sorted(dir_stats.items(), key=lambda x: x[1]['size'], reverse=True):
        print(f"\nDirectory: {dir_path}")
        print(f"Files: {stats['files']}")
        print(f"Total Size: {stats['size'] / 1024 / 1024:.2f} MB")

def _show_duplicate_groups(groups: Mapping[str, GroupType]) -> None:
    """Show detailed information about duplicate groups"""
    for name, group_data in groups.items():
        files = group_data['files'] if isinstance(group_data, dict) else group_data
        print(f"\nGroup: {name}")
        print("-" * 80)
        
        # Group files by directory
        dir_groups: Dict[str, List[FileInfo]] = defaultdict(list)
        for file in files:
            file_info = FileInfo(
                name=file['name'],
                path=file['path'],
                size=file['size'],
                original_name=file.get('original_name'),
                book_info=file.get('book_info')
            )
            dir_path = str(Path(file_info['path']).parent)
            dir_groups[dir_path].append(file_info)
            
            for dir_path, dir_files in dir_groups.items():
                print(f"\nDirectory: {dir_path}")
                dir_size = sum(f['size'] for f in dir_files)
                print(f"Directory Size: {dir_size / 1024 / 1024:.2f} MB")
                
                for file in dir_files:
                    print(f"  - {file['name']} ({file['size'] / 1024 / 1024:.2f} MB)")

def _show_file_patterns(groups: Mapping[str, GroupType]) -> None:
    """Analyze and show file naming patterns"""
    patterns = defaultdict(int)
    
    for group_data in groups.values():
        files = group_data['files'] if isinstance(group_data, dict) else group_data
        for file in files:
            # Only convert if it's a dict, not if it's already FileInfo
            if not isinstance(file, dict):
                file_info = file
            else:
                file_info = convert_to_fileinfo(file)
            pattern = re.sub(r'\d+', '#', file_info['name'])
            pattern = re.sub(r'[a-zA-Z]+', 'W', pattern)
            patterns[pattern] += 1
    
    print("\nCommon File Patterns:")
    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{pattern}: {count} occurrences")

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    logger.info("\nReceived interrupt signal. Forcing immediate exit...")
    # Force immediate exit without waiting
    if sys.platform != "win32":
        os.killpg(os.getpgid(0), signal.SIGKILL)
    else:
        os.kill(os.getpid(), signal.SIGKILL)  # Use SIGKILL instead of SIGTERM

# Register handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main() -> None:
    # Create DuplicateFinder instance
    finder = DuplicateFinder()
    
    # Get directories to scan
    directories = get_directories()
    if not directories:
        logger.error("No directories provided. Exiting.")
        return
        
    try:
        # Process directories with default extensions
        finder.process_directories(directories, SUPPORTED_EXTENSIONS)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error processing directories: {e}")

if __name__ == "__main__":
    main()

def generate_html_report(groups: Mapping[str, GroupType], timestamp: str) -> str:
    """Generate a detailed HTML report of duplicate groups"""
    report_dir = "Reports"
    os.makedirs(report_dir, exist_ok=True)
    
    html_report = os.path.join(report_dir, f"duplicate_report_{timestamp}.html")
    
    css = """
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
            .group { background: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .group-header { background: #34495e; color: white; padding: 10px; margin: -20px -20px 20px; border-radius: 5px 5px 0 0; }
            .metadata { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 3px; }
            .directory { margin: 15px 0; padding: 10px; background: #f9f9f9; border-left: 4px solid #2980b9; }
            .file { margin: 5px 0 5px 20px; color: #444; }
            .stats { font-weight: bold; color: #2c3e50; }
            .total { background: #27ae60; color: white; padding: 15px; margin-top: 20px; border-radius: 5px; }
        </style>
    """
    
    with open(html_report, 'w', encoding='utf-8') as f:
        f.write(f"""
        <html>
        <head>
            <title>Duplicate Files Report - {timestamp}</title>
            {css}
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Duplicate Files Report</h1>
                    <p>Scan Date: {timestamp}</p>
                    <p>Total Groups: {len(groups)}</p>
                </div>
        """)
        
        total_savings = 0
        for normalized, group_data in groups.items():
            if isinstance(group_data, dict) and 'files' in group_data:
                files = group_data['files']
                metadata = group_data.get('metadata')
                suggested_name = group_data.get('suggested_name')
            else:
                files = group_data
                metadata = None
                suggested_name = None
            
            if len(files) > 1:
                total_size = sum(file['size'] for file in files)
                potential_savings = total_size - min(file['size'] for file in files)
                total_savings += potential_savings
                
                f.write(f"""
                    <div class="group">
                        <div class="group-header">
                            <h2>{normalized}</h2>
                        </div>
                """)
                
                if metadata:
                    f.write(f"""
                        <div class="metadata">
                            <h3>Book Information:</h3>
                            <p>Title: {metadata.title}</p>
                            <p>Author: {', '.join(metadata.authors)}</p>
                            <p>Published: {metadata.publish_year or 'Unknown'}</p>
                            <p>ISBN: {metadata.isbn or 'Unknown'}</p>
                            {'<p>Series: ' + metadata.series_name + ' #' + str(metadata.series_index) + '</p>' if metadata.series_name else ''}
                            <p>Suggested Name: {suggested_name}</p>
                        </div>
                    """)
                
                # Group files by directory
                dir_groups = defaultdict(list)
                for file in files:
                    dir_path = str(Path(file['path']).parent)
                    dir_groups[dir_path].append(file)
                
                for dir_path, dir_files in dir_groups.items():
                    dir_size = sum(f['size'] for f in dir_files)
                    f.write(f"""
                        <div class="directory">
                            <h4>Directory: {dir_path}</h4>
                            <p>Size: {format_size(dir_size)}</p>
                    """)
                    
                    for file in dir_files:
                        f.write(f"""
                            <div class="file">
                                {file['name']} ({format_size(file['size'])})
                            </div>
                        """)
                    f.write("</div>")
                f.write("</div>")
        
        f.write(f"""
                <div class="total">
                    <h2>Total Potential Space Savings: {format_size(total_savings)}</h2>
                </div>
            </div>
        </body>
        </html>
        """)
    
    return html_report

def save_potential_duplicates(groups: Mapping[str, GroupType], timestamp: str) -> str:
    """Save potential duplicate groups before OpenLibrary verification"""
    report_dir = "Reports"
    os.makedirs(report_dir, exist_ok=True)
    
    output_file = os.path.join(report_dir, f"potential_duplicates_{timestamp}.json")
    
    # Convert to serializable format
    serializable_groups: Dict[str, List[Dict[str, Any]]] = {}
    for name, group_data in groups.items():
        files = group_data['files'] if isinstance(group_data, dict) else group_data
        serializable_files = []
        for file in files:
            # Cast to ensure type safety
            file_info = cast(FileInfo, {
                'name': file['name'],
                'path': str(file['path']),
                'size': int(file['size']),
                'original_name': file.get('original_name'),
                'book_info': file.get('book_info')
            })
            serializable_files.append(dict(file_info))
        serializable_groups[name] = serializable_files
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_groups, f, indent=2, ensure_ascii=False)
    
    return output_file

@overload
def convert_to_fileinfo(file_dict: Dict[str, Any]) -> FileInfo: ...

@overload
def convert_to_fileinfo(file_dict: FileInfo) -> FileInfo: ...

def convert_to_fileinfo(file_dict: Union[Dict[str, Any], FileInfo]) -> FileInfo:
    """Convert a dictionary to FileInfo TypedDict or return existing FileInfo"""
    if isinstance(file_dict, dict):
        return FileInfo(
            name=str(file_dict['name']),
            path=str(file_dict['path']),
            size=int(file_dict['size']),
            original_name=file_dict.get('original_name'),
            book_info=file_dict.get('book_info')
        )
    return file_dict

def convert_groups(groups: InitialGroupType) -> Dict[str, GroupType]:
    """Convert initial groups to properly typed groups"""
    return {
        name: [convert_to_fileinfo(f) for f in files]
        for name, files in groups.items()
    }

@overload
def ensure_fileinfo(file: Dict[str, Any]) -> FileInfo: ...

@overload
def ensure_fileinfo(file: FileInfo) -> FileInfo: ...

def ensure_fileinfo(file: Union[Dict[str, Any], FileInfo]) -> FileInfo:
    """Ensure a file is in FileInfo format"""
    if isinstance(file, dict):
        return FileInfo(
            name=str(file['name']),
            path=str(file['path']),
            size=int(file['size']),
            original_name=file.get('original_name'),
            book_info=file.get('book_info')
        )
    return file

if __name__ == "__main__":
    main()

class OpenLibraryRateLimiter:
    def __init__(self, requests_per_second: float = 1.0):
        self.rate_limit = requests_per_second
        self.last_request = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request
            if time_since_last < (1.0 / self.rate_limit):
                sleep_time = (1.0 / self.rate_limit) - time_since_last
                time.sleep(sleep_time)
            self.last_request = time.time()

class APIStats:
    def __init__(self):
        self.requests = 0
        self.timeouts = 0
        self.successes = 0
        self.failures = 0
        self.start_time = time.time()
        
    def log_timeout(self):
        self.timeouts += 1
        self.requests += 1
        
    def log_success(self):
        self.successes += 1
        self.requests += 1
        
    def log_failure(self):
        self.failures += 1
        self.requests += 1
        
    def get_stats(self) -> Dict[str, float]:
        elapsed = time.time() - self.start_time
        return {
            'success_rate': (self.successes / self.requests * 100) if self.requests else 0,
            'timeout_rate': (self.timeouts / self.requests * 100) if self.requests else 0,
            'requests_per_second': self.requests / elapsed if elapsed > 0 else 0
        }

class RetryConfig:
    def __init__(self, 
                 max_retries: int = 3,
                 initial_delay: float = 1.0,
                 max_delay: float = 30.0,
                 timeout: int = 10):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.timeout = timeout