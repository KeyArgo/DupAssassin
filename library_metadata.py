from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple
import re
from datetime import datetime
import os

@dataclass
class BookMetadata:
    title: str
    authors: List[str]
    isbn: Optional[str] = None
    publish_year: Optional[int] = None
    edition_count: Optional[int] = None
    series_name: Optional[str] = None
    series_index: Optional[float] = None
    edition_key: Optional[str] = None
    ol_key: Optional[str] = None
    filepath: Optional[str] = None

class LibraryMetadataProcessor:
    def __init__(self):
        self.cache = {}  # Consider implementing persistent cache
        
    def extract_series_info(self, title: str) -> 'Tuple[Optional[str], Optional[float]]':
        """Extract series name and index from title"""
        series_patterns = [
            r'(?i)(.+?)\s*(?:book|volume|vol|v|#)?\s*(\d+(?:\.\d+)?)',
            r'(?i)(.+?)\s*\(.*?book\s*(\d+(?:\.\d+)?)',
        ]
        
        for pattern in series_patterns:
            if match := re.search(pattern, title):
                series_name = match.group(1).strip()
                try:
                    series_index = float(match.group(2))
                    return series_name, series_index
                except (ValueError, TypeError):
                    pass
        return None, None

    def normalize_book_data(self, ol_data: Dict, filepath: Optional[str] = None) -> BookMetadata:
        """Convert OpenLibrary data to standardized BookMetadata"""
        title = ol_data.get('title', 'Unknown')
        series_name, series_index = self.extract_series_info(title)
        
        # Ensure series_index is float or None
        if series_index is not None:
            try:
                series_index = float(series_index)
            except (ValueError, TypeError):
                series_index = None
        
        return BookMetadata(
            title=title,
            authors=ol_data.get('author_name', []),
            isbn=ol_data.get('isbn', [None])[0],
            publish_year=self._extract_year(ol_data.get('first_publish_date')),
            edition_count=len(ol_data.get('edition_key', [])),
            series_name=series_name,
            series_index=series_index,
            edition_key=ol_data.get('edition_key', [None])[0],
            ol_key=ol_data.get('key'),
            filepath=filepath
        )

    def generate_filename(self, metadata: BookMetadata) -> str:
        """Generate standardized filename from metadata"""
        components = []
        
        # Add series info if available
        if metadata.series_name and metadata.series_index:
            components.append(f"{metadata.series_name} - {metadata.series_index:02.1f}")
        
        # Add title
        components.append(metadata.title)
        
        # Add author
        if metadata.authors:
            components.append(f"by {metadata.authors[0]}")
            
        # Add year if available
        if metadata.publish_year:
            components.append(f"({metadata.publish_year})")
            
        # Clean and join components
        filename = " - ".join(components)
        return self._sanitize_filename(filename)

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Clean filename of invalid characters"""
        invalid_chars = r'[<>:"/\\|?*]'
        return re.sub(invalid_chars, '', filename)

    @staticmethod
    def _extract_year(date_str: Optional[str]) -> Optional[int]:
        """Extract year from date string"""
        if not date_str:
            return None
        match = re.search(r'\d{4}', date_str)
        return int(match.group()) if match else None
