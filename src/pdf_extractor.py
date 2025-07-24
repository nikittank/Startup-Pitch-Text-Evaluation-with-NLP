"""
PDF text extraction using pdfplumber for robust text extraction.
"""
import logging
import pdfplumber
from typing import Dict, List, Any, Optional
from pathlib import Path
import re

from ..models.interfaces import PDFExtractorInterface


class PDFExtractor(PDFExtractorInterface):
    """PDF text extractor using pdfplumber for robust extraction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract raw text content from PDF pitch deck.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text, metadata, and quality metrics
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            result = {
                'text': '',
                'page_count': 0,
                'extraction_success': True,
                'error_message': None,
                'quality_score': 0.0
            }
            
            with pdfplumber.open(pdf_path) as pdf:
                result['page_count'] = len(pdf.pages)
                all_text = []
                successful_pages = 0
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            all_text.append(page_text)
                            successful_pages += 1
                        else:
                            self.logger.warning(f"No text extracted from page {page_num} in {pdf_path}")
                    except Exception as e:
                        self.logger.error(f"Error extracting text from page {page_num}: {e}")
                        continue
                
                result['text'] = '\n\n'.join(all_text)
                result['quality_score'] = successful_pages / result['page_count'] if result['page_count'] > 0 else 0.0
                
                if not result['text'].strip():
                    # Check if this is an image-based PDF
                    is_image_based = self._detect_image_based_pdf(pdf_path)
                    result['extraction_success'] = False
                    if is_image_based:
                        result['error_message'] = "PDF appears to be image-based - OCR would be required for text extraction"
                    else:
                        result['error_message'] = "No text content extracted from PDF"
                    
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            result = {
                'text': '',
                'page_count': 0,
                'extraction_success': False,
                'error_message': str(e),
                'quality_score': 0.0
            }
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {e}")
            result = {
                'text': '',
                'page_count': 0,
                'extraction_success': False,
                'error_message': f"PDF processing error: {str(e)}",
                'quality_score': 0.0
            }
            
        return result
    
    def extract_slides(self, pdf_path: str) -> List[str]:
        """
        Extract text content slide by slide for structured processing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of strings, each representing text from one slide/page
        """
        slides = []
        
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            slides.append(page_text.strip())
                        else:
                            # Add empty string for pages with no text to maintain slide numbering
                            slides.append("")
                            self.logger.warning(f"No text found on slide {page_num}")
                    except Exception as e:
                        self.logger.error(f"Error extracting text from slide {page_num}: {e}")
                        slides.append("")  # Maintain slide numbering
                        
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
        except Exception as e:
            self.logger.error(f"Error processing PDF slides {pdf_path}: {e}")
            
        return slides
    
    def get_metadata(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract metadata from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        metadata = {
            'title': '',
            'author': '',
            'subject': '',
            'creator': '',
            'producer': '',
            'creation_date': '',
            'modification_date': '',
            'file_size': '',
            'page_count': '0'
        }
        
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            # Get file size
            metadata['file_size'] = str(pdf_path.stat().st_size)
            
            with pdfplumber.open(pdf_path) as pdf:
                metadata['page_count'] = str(len(pdf.pages))
                
                # Extract PDF metadata
                pdf_metadata = pdf.metadata or {}
                
                # Map PDF metadata fields to our metadata structure
                metadata_mapping = {
                    'title': ['Title', '/Title'],
                    'author': ['Author', '/Author'],
                    'subject': ['Subject', '/Subject'],
                    'creator': ['Creator', '/Creator'],
                    'producer': ['Producer', '/Producer'],
                    'creation_date': ['CreationDate', '/CreationDate'],
                    'modification_date': ['ModDate', '/ModDate']
                }
                
                for key, possible_keys in metadata_mapping.items():
                    for pdf_key in possible_keys:
                        if pdf_key in pdf_metadata and pdf_metadata[pdf_key]:
                            metadata[key] = str(pdf_metadata[pdf_key])
                            break
                            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            
        return metadata
    
    def _handle_password_protected(self, pdf_path: str, common_passwords: List[str] = None) -> Optional[str]:
        """
        Attempt to handle password-protected PDFs with common passwords.
        
        Args:
            pdf_path: Path to the PDF file
            common_passwords: List of common passwords to try
            
        Returns:
            Password if successful, None otherwise
        """
        if common_passwords is None:
            common_passwords = ['', 'password', '123456', 'admin', 'user']
            
        for password in common_passwords:
            try:
                with pdfplumber.open(pdf_path, password=password) as pdf:
                    # Try to access first page to verify password works
                    if len(pdf.pages) > 0:
                        pdf.pages[0].extract_text()
                        self.logger.info(f"Successfully opened password-protected PDF with password")
                        return password
            except Exception:
                continue
                
        self.logger.warning(f"Could not open password-protected PDF: {pdf_path}")
        return None
    
    def _detect_image_based_pdf(self, pdf_path: str) -> bool:
        """
        Detect if PDF is primarily image-based (no extractable text).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF appears to be image-based
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                pages_with_images = 0
                pages_with_text = 0
                
                # Sample first few pages to determine PDF type
                sample_pages = min(3, total_pages)
                
                for i in range(sample_pages):
                    page = pdf.pages[i]
                    
                    # Check for text content
                    text = page.extract_text()
                    if text and text.strip():
                        pages_with_text += 1
                    
                    # Check for images
                    if hasattr(page, 'images') and page.images:
                        pages_with_images += 1
                
                # If most pages have images but no text, likely image-based
                return pages_with_images > 0 and pages_with_text == 0
                
        except Exception as e:
            self.logger.error(f"Error detecting PDF type: {e}")
            return False
    
    def validate_extraction_quality(self, extracted_text: str, expected_min_length: int = 100) -> Dict[str, Any]:
        """
        Assess the quality of text extraction.
        
        Args:
            extracted_text: The extracted text content
            expected_min_length: Minimum expected text length
            
        Returns:
            Dictionary with quality assessment metrics
        """
        quality_metrics = {
            'text_length': len(extracted_text),
            'word_count': len(extracted_text.split()) if extracted_text else 0,
            'has_minimum_content': len(extracted_text) >= expected_min_length,
            'contains_common_artifacts': False,
            'estimated_completeness': 0.0
        }
        
        # Check for common PDF extraction artifacts
        artifacts = [
            r'\x0c',  # Form feed characters
            r'[^\x00-\x7F]+',  # Non-ASCII characters that might indicate encoding issues
            r'\.{10,}',  # Long sequences of dots
            r'\s{10,}',  # Excessive whitespace
        ]
        
        for artifact_pattern in artifacts:
            if re.search(artifact_pattern, extracted_text):
                quality_metrics['contains_common_artifacts'] = True
                break
        
        # Estimate completeness based on text characteristics
        if quality_metrics['word_count'] > 0:
            # Simple heuristic: longer text with varied vocabulary suggests better extraction
            unique_words = len(set(extracted_text.lower().split()))
            vocabulary_ratio = unique_words / quality_metrics['word_count']
            length_score = min(quality_metrics['text_length'] / 1000, 1.0)  # Normalize to 1000 chars
            
            quality_metrics['estimated_completeness'] = (vocabulary_ratio * 0.6 + length_score * 0.4)
        
        return quality_metrics