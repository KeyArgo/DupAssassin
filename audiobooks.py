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
from typing import Dict, List, Set, Optional, TypedDict, Sequence, Any, Tuple, Union, Mapping
from urllib.parse import quote
from functools import lru_cache
import gc
import signal
import sys
import threading
from concurrent.futures import wait

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
    log_dir = Path('logs')
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

class FileInfo(TypedDict):
    name: str
    path: str
    size: int
    original_name: Optional[str]
    book_info: Optional[BookInfo]

class DuplicateGroup(TypedDict):
    files: List[FileInfo]
    ol_info: Optional[Dict[str, Any]]

GroupType = Union[List[FileInfo], DuplicateGroup]

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
                    rate = processed_count / (current_time - start_time)
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

def search_openlibrary(book_info: BookInfo) -> Optional[Dict[str, Any]]:
    """Modified to accept BookInfo type"""
    query = {
        'title': book_info['title'],
        'author': book_info['author']
    }
    if book_info['series_info']['series']:
        query['series'] = book_info['series_info']['series']
    
    return search_ol_api(query)  # Assuming this is your actual API call function

def group_by_directory(files):
    """Group files by their parent directory"""
    groups = defaultdict(list)
    for f in files:
        parent = str(Path(f['path']).parent)
        groups[parent].append(f)
    return groups

def print_duplicates(duplicates, include_ol_info=False, output_file=None):
    """Print duplicate files in a readable format"""
    # Generate timestamp for the report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = "Reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # Always create a report file if one isn't specified
    if not output_file:
        output_file = os.path.join(report_dir, f"duplicates_{timestamp}.txt")
    
    output = []
    if not duplicates:
        output.append("\nNo duplicates found.")
    else:
        output.append("\nDuplicate Groups Found:")
        output.append("=" * 80)
        total_wasted_space = 0
        
        for name, group in sorted(duplicates.items()):
            files = group.get('files', group) if include_ol_info else group
            
            if include_ol_info and isinstance(group, dict) and 'ol_info' in group:
                ol_info = group['ol_info']
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
                        print("Missing 'name' key in:", f)
                for f in sorted(dir_files, key=lambda x: x.get('name', 'Unknown Name')):
                    name = f.get('name', 'Unknown Name')
                    output.append(f"    - {name}")
            output.append("\n" + "=" * 80)
        output.append("\n" + "=" * 80)
        output.append(f"Total Potential Space Savings: {total_wasted_space / (1024**3):.2f} GB")
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output))
    else:
        print('\n'.join(output))

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
        batch_size = 100  # Smaller batch size
        processed_count = 0
        
        try:
            with tqdm(total=len(files), desc="Normalizing files", unit="files") as pbar:
                for i in range(0, len(files), batch_size):
                    if self.check_interrupt():
                        return groups
                    
                    batch = files[i:i + batch_size]
                    
                    # Log problematic files
                    if 0.47 <= (i / len(files)) <= 0.49:
                        for file in batch:
                            start_time = time.time()
                            normalized = normalize_filename(file['path'], len(files))
                            duration = time.time() - start_time
                            if duration > 0.1:  # Log slow normalizations
                                logger.warning(f"Slow normalization ({duration:.2f}s): {file['path']}")
                            groups[normalized].append(file)
                            processed_count += 1
                            pbar.update(1)
                    else:
                        # Normal batch processing
                        for file in batch:
                            normalized = normalize_filename(file['path'], len(files))
                            groups[normalized].append(file)
                            processed_count += 1
                            pbar.update(1)
                    
                    # More frequent garbage collection
                    if (i // batch_size) % 5 == 0:
                        gc.collect()
        
        except Exception as e:
            logger.error(f"Error in normalization: {e}")
            
        return groups

    def process_directories(self, directories: List[str], extensions: List[str]) -> Dict[str, DuplicateGroup]:
        """Process directories with explicit phase transitions"""
        try:
            # Phase 1: Initial file scan
            self.stats['scan_phase'] = 'scanning'
            self.stats['last_progress_time'] = time.time()
            logger.info("Starting initial file scan...")
            all_files = self._scan_directories(directories, extensions)
            
            if not all_files:
                logger.warning("No files found matching the specified extensions")
                return {}
                
            # Phase 2: File normalization and grouping
            self.stats['scan_phase'] = 'normalizing'
            self.stats['last_progress_time'] = time.time()
            logger.info(f"Starting normalization of {len(all_files)} files...")
            initial_groups = self._normalize_and_group(all_files)
            
            if not initial_groups:
                logger.warning("No groups formed after normalization")
                return {}
                
            # Phase 3: Analysis
            self.stats['scan_phase'] = 'analyzing'
            logger.info(f"Starting analysis of {len(initial_groups)} groups...")
            
            # Show preview before proceeding with analysis
            if not preview_initial_scan(self, initial_groups):
                logger.info("User cancelled processing")
                return {}
            
            # Process with OpenLibrary API
            logger.info("Starting OpenLibrary API processing...")
            enriched_groups = process_with_openlibrary(initial_groups)
            
            # Generate final report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    def generate_report(self, groups: Mapping[str, GroupType], timestamp: str) -> str:
        """Generate a detailed report of duplicate groups"""
        report_dir = "Reports"
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(report_dir, f"duplicate_report_{timestamp}.txt")
        html_report = os.path.join(report_dir, f"duplicate_report_{timestamp}.html")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== Duplicate Files Report ===\n\n")
            f.write(f"Scan Date: {timestamp}\n")
            f.write(f"Total Groups: {len(groups)}\n\n")
            
            for normalized, group_data in groups.items():
                if isinstance(group_data, dict) and 'files' in group_data:
                    files = group_data['files']
                    ol_info = group_data.get('ol_info')
                else:
                    files = group_data
                    ol_info = None
                
                if len(files) > 1:  # Only report actual duplicates
                    f.write(f"\nGroup: {normalized}\n")
                    if ol_info:
                        f.write(f"Title: {ol_info.get('title', 'Unknown')}\n")
                        f.write(f"Author: {ol_info.get('author', 'Unknown')}\n")
                    
                    total_size = sum(file['size'] for file in files)
                    f.write(f"Total Group Size: {total_size / (1024*1024):.2f} MB\n")
                    
                    for file in files:
                        f.write(f"  {file['path']} ({file['size'] / (1024*1024):.2f} MB)\n")
                    f.write("-" * 80 + "\n")
        
        # Generate HTML report for better visualization
        with open(html_report, 'w', encoding='utf-8') as f:
            f.write("""
            <html><head><style>
                body { font-family: Arial; margin: 20px; }
                .group { margin: 20px 0; padding: 10px; border: 1px solid #ccc; }
                .size { color: #666; }
            </style></head><body>
            """)
            # Add HTML content similar to text report
            f.write("</body></html>")
        
        return report_file

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

def scan_directories(directories: List[str], extensions: List[str] = SUPPORTED_EXTENSIONS) -> Tuple[List[Dict[str, Any]], int]:
    """Scan directories for audio files and return files list and total count."""
    all_files = []
    total_files = 0
    
    # First count total files
    for directory in directories:
        for root, _, files in os.walk(directory):
            total_files += len([f for f in files if f.endswith(tuple(extensions))])
    
    # Now scan with progress bar
    with tqdm(total=total_files, desc="Scanning files", unit="files") as pbar:
        for directory in directories:
            try:
                for root, _, files in os.walk(directory):
                    matching_files = [
                        {
                            'name': f,
                            'path': os.path.join(root, f),
                            'size': os.path.getsize(os.path.join(root, f))
                        }
                        for f in files if f.endswith(tuple(extensions))
                    ]
                    all_files.extend(matching_files)
                    pbar.update(len(matching_files))
            except Exception as e:
                logger.error(f"Error scanning directory {directory}: {e}")
                continue
    
    return all_files, total_files

def process_with_openlibrary(duplicates: Mapping[str, List[FileInfo]], batch_size: int = 10) -> Dict[str, DuplicateGroup]:
    """Process files with OpenLibrary API with progress tracking."""
    results: Dict[str, DuplicateGroup] = {}
    total_groups = len(duplicates)
    
    with tqdm(total=total_groups, desc="OpenLibrary API lookup", unit="groups") as pbar:
        for group_name, files in duplicates.items():
            try:
                # Use first file in group for OpenLibrary lookup
                if files:
                    parent_dir = os.path.dirname(files[0]['path'])
                    book_name = os.path.basename(parent_dir)
                    book_info = extract_book_info(book_name, parent_dir)
                    ol_info = search_openlibrary(book_info)
                    
                    results[group_name] = {
                        'files': files,
                        'ol_info': ol_info if ol_info else None
                    }
                
                pbar.update(1)
                sleep(1)  # Respect API rate limits
                
            except Exception as e:
                logger.error(f"Error processing group {group_name}: {e}")
                continue
            
            pbar.set_postfix({
                'success_rate': f'{len(results)}/{total_groups} ({len(results)/total_groups:.1%})'
            })
    
    return results

def preview_duplicates(duplicates: Mapping[str, GroupType], finder: DuplicateFinder) -> bool:
    """Preview duplicate groups before processing with OpenLibrary"""
    print(f"\nFound {len(duplicates)} potential duplicate groups")
    
    while True:
        print("\nOptions:")
        print("1. View all duplicate groups")
        print("2. View summary statistics")
        print("3. View largest duplicate groups")
        print("4. Continue with OpenLibrary processing")
        print("5. Exit")
        print("6. Show processing status")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            for name, group in duplicates.items():
                print(f"\nGroup: {name}")
                print("-" * 80)
                files = group['files'] if isinstance(group, dict) else group
                for f in files:
                    print(f"  {f['path']}")
                    print(f"  Size: {f['size'] / 1024 / 1024:.2f} MB")
                print()
                
        elif choice == '2':
            total_groups = len(duplicates)
            total_files = sum(len(group['files'] if isinstance(group, dict) else group) 
                            for group in duplicates.values())
            total_size = sum(sum(f['size'] for f in (group['files'] if isinstance(group, dict) else group))
                           for group in duplicates.values())
            
            print(f"\nTotal duplicate groups: {total_groups}")
            print(f"Total duplicate files: {total_files}")
            print(f"Total size of duplicates: {total_size / 1024 / 1024 / 1024:.2f} GB")
            
        elif choice == '3':
            sorted_groups = sorted(
                duplicates.items(),
                key=lambda x: sum(f['size'] for f in (x[1]['files'] if isinstance(x[1], dict) else x[1])),
                reverse=True
            )[:10]
            
            for name, group in sorted_groups:
                files = group['files'] if isinstance(group, dict) else group
                total_size = sum(f['size'] for f in files)
                print(f"\nGroup: {name}")
                print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
                print(f"Number of duplicates: {len(files)}")
                
        elif choice == '4':
            return True
            
        elif choice == '5':
            return False
            
        elif choice == '6':
            print("\nProcessing Status:")
            print(f"Files processed: {finder.stats['files_processed']}")
            print(f"Processing rate: {finder.stats['files_processed'] / (time.time() - finder.stats['start_time']):.1f} files/sec")
            print(f"Time since last progress: {time.time() - finder.stats['last_progress_time']:.1f} seconds")

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

def _show_directory_stats(groups: Dict[str, List[FileInfo]]) -> None:
    """Show statistics about directories containing duplicates"""
    dir_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'files': 0, 'size': 0})
    
    for files in groups.values():
        for file in files:
            dir_path = str(Path(file['path']).parent)
            dir_stats[dir_path]['files'] += 1
            dir_stats[dir_path]['size'] += file['size']
    
    print("\nDirectory Statistics:")
    for dir_path, stats in sorted(dir_stats.items(), key=lambda x: x[1]['size'], reverse=True):
        print(f"\nDirectory: {dir_path}")
        print(f"Files: {stats['files']}")
        print(f"Total Size: {stats['size'] / 1024 / 1024:.2f} MB")

def _show_duplicate_groups(groups: Dict[str, List[FileInfo]]) -> None:
    """Show detailed information about duplicate groups"""
    for name, files in groups.items():
        print(f"\nGroup: {name}")
        print("-" * 80)
        
        # Group files by directory
        dir_groups: Dict[str, List[FileInfo]] = defaultdict(list)
        for file in files:
            dir_path = str(Path(file['path']).parent)
            dir_groups[dir_path].append(file)
        
        # Show files grouped by directory
        for dir_path, dir_files in dir_groups.items():
            print(f"\nDirectory: {dir_path}")
            dir_size = sum(f['size'] for f in dir_files)
            print(f"Directory Size: {dir_size / 1024 / 1024:.2f} MB")
            
            for file in dir_files:
                print(f"  - {file['name']} ({file['size'] / 1024 / 1024:.2f} MB)")

def _show_file_patterns(groups: Dict[str, List[FileInfo]]) -> None:
    """Analyze and show file naming patterns"""
    patterns = defaultdict(int)
    
    for files in groups.values():
        for file in files:
            # Extract pattern from filename
            pattern = re.sub(r'\d+', '#', file['name'])
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
        logger.error("No valid directories provided.")
        return
    
    # Initial scan and grouping
    duplicates = finder.process_directories(directories, SUPPORTED_EXTENSIONS)
    
    if duplicates:
        # Preview initial results
        if preview_initial_scan(finder, duplicates):
            # Continue with OpenLibrary processing
            if preview_duplicates(duplicates, finder):
                print_duplicates(duplicates, include_ol_info=True)
            else:
                print_duplicates(duplicates, include_ol_info=False)
        else:
            print_duplicates(duplicates, include_ol_info=False)

if __name__ == "__main__":
    main()