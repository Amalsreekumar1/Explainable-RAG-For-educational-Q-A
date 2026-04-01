import sys
import os
import requests
import shutil
import base64
import logging
import hashlib
import gc
import tempfile
from io import BytesIO
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import re
import fitz  # Only used for safely splitting PDFs, NOT for text extraction
from PIL import Image
import imagehash
from tqdm import tqdm

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    HTMLFormatOption,
    PowerpointFormatOption
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import TextItem, TableItem, PictureItem, DocItemLabel

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 25  # Process PDFs 25 pages at a time to prevent RAM crashes

# ==================== VLM Singleton with Anti-Hallucination ====================
class VLMProcessor:
    """Singleton VLM client with generation limits to prevent repetition loops."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        try:
            from langchain_ollama import ChatOllama
            
            self.llm = ChatOllama(
                model="llava-phi3", 
                temperature=0.1,
                max_tokens=150,           # Force stop after short paragraph
                repeat_penalty=1.5,       # Heavily penalize repetition
                top_p=0.9                 # Keep focused and factual
            )
            self._initialized = True
            logger.info("VLM initialized with anti-hallucination safeguards")
        except Exception as e:
            logger.error(f"Failed to initialize VLM: {e}")
            self.llm = None
    
    def describe(self, image_path: Path, source_name: str) -> str:
        """Generate description for image with safety checks"""
        if not self.llm:
            return f"[Visual content: {image_path.name} from {source_name}]"
        
        try:
            from langchain_core.messages import HumanMessage
            
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this educational diagram in 1-2 sentences. Be specific and concise. If it is just a logo or decorative, reply exactly with 'IGNORE_IMAGE'."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            )
            
            response = self.llm.invoke([message])
            description = response.content.strip()
            
            if "IGNORE_IMAGE" in description:
                return "IGNORE_IMAGE"
                
            description = self._sanitize_description(description)
            return f"[Diagram/Visual: {description}]"
            
        except Exception as e:
            logger.error(f"VLM description failed for {image_path.name}: {e}")
            return f"[Visual content: {image_path.name} from {source_name}]"
    
    def _sanitize_description(self, text: str) -> str:
        """Emergency cleanup for repetition loops."""
        if not text:
            return text
            
        lines = text.split('. ')
        if len(lines) > 3:
            first_part = ' '.join(lines[:2]).lower()
            last_part = ' '.join(lines[-2:]).lower()
            
            first_words = set(first_part.split())
            last_words = set(last_part.split())
            if len(first_words) > 0:
                overlap = len(first_words & last_words) / len(first_words)
                if overlap > 0.7:  # 70% same words = repetition
                    logger.warning(f"Detected repetition loop, truncating")
                    return '. '.join(lines[:3]) + "."
        
        if len(text) > 800:
            text = text[:800].rsplit('.', 1)[0] + "."
            
        return text

_vlm_processor = None

def get_vlm():
    global _vlm_processor
    if _vlm_processor is None:
        _vlm_processor = VLMProcessor()
    return _vlm_processor

# ==================== Perceptual Hashing ====================
def get_perceptual_hash(img: Image.Image) -> str:
    return str(imagehash.phash(img))

def is_valid_image(img: Image.Image, seen_hashes: set) -> Tuple[bool, str]:
    w, h = img.size
    if w < 200 or h < 150:
        return False, ""
    
    ratio = max(w, h) / min(w, h)
    if ratio > 5:
        return False, ""
    
    img_hash = get_perceptual_hash(img)
    if img_hash in seen_hashes:
        return False, img_hash
    
    return True, img_hash

# ==================== Incremental CSV Writing ====================
class KnowledgeBaseWriter:
    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.required_cols = ["chunk_id", "source_file", "page_number", "text", 
                              "chunk_length", "type", "image_ref", "section_path", 
                              "image_description"]
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        if not os.path.exists(self.kb_path):
            pd.DataFrame(columns=self.required_cols).to_csv(self.kb_path, index=False)
            logger.info(f"Created new KB: {self.kb_path}")
    
    def get_existing_sources(self) -> set:
        if not os.path.exists(self.kb_path): return set()
        try:
            return set(pd.read_csv(self.kb_path, usecols=['source_file'])['source_file'].unique())
        except: return set()
    
    def get_visual_asset_count(self) -> int:
        if not os.path.exists(self.kb_path): return 0
        try:
            return len(pd.read_csv(self.kb_path, usecols=['type']).query("type == 'visual_content'"))
        except: return 0
    
    def append_chunks(self, chunks: List[Dict]):
        if not chunks: return
        df_new = pd.DataFrame(chunks)
        for col in self.required_cols:
            if col not in df_new.columns: df_new[col] = ""
        df_new = df_new[self.required_cols]
        write_header = not os.path.exists(self.kb_path) or os.path.getsize(self.kb_path) == 0
        df_new.to_csv(self.kb_path, mode='a', header=write_header, index=False)
        logger.info(f"Appended {len(chunks)} chunks to KB")


# ==================== 100% Docling PDF Chunking Logic ====================
def split_pdf_to_chunks(source: str, output_dir: Path) -> List[Tuple[str, int]]:
    """Split large PDF into chunks using fitz (PyMuPDF) only for page slicing."""
    doc = fitz.open(source)
    total_pages = len(doc)
    chunks = []
    
    logger.info(f"Splitting {total_pages} pages into {CHUNK_SIZE}-page chunks...")
    
    for start in range(0, total_pages, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_pages)
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start, to_page=end-1)
        
        chunk_path = output_dir / f"chunk_{start+1}_{end}.pdf"
        chunk_doc.save(str(chunk_path))
        chunk_doc.close()
        
        chunks.append((str(chunk_path), start))  # (path, page_offset)
    
    doc.close()
    return chunks

def process_chunk(chunk_path: str, page_offset: int, source_name: str, assets_dir: Path, 
                  counter: int, vlm_enabled: bool, seen_hashes: set, chunk_counter: int) -> Tuple[List[Dict], int, int]:
    """Process one PDF chunk with Docling (extracts text, tables, images)."""
    pdf_opt = PdfPipelineOptions()
    pdf_opt.do_ocr = False 
    pdf_opt.do_table_structure = True
    pdf_opt.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opt)}
    )
    
    try:
        doc = converter.convert(chunk_path).document
        chunks = []
        current_section = "General"
        vlm = get_vlm() if vlm_enabled else None
        
        for element, level in doc.iterate_items():
            actual_page = 1
            if hasattr(element, "prov") and element.prov:
                actual_page = element.prov[0].page_no + page_offset
            
            if isinstance(element, TextItem):
                if element.label == DocItemLabel.SECTION_HEADER:
                    current_section = element.text.strip()
                    continue 
            
            # Text/Tables
            if isinstance(element, (TextItem, TableItem)):
                content = element.export_to_markdown() if isinstance(element, TableItem) else str(element.text)
                content = re.sub(r'\s+', ' ', content).strip()
                
                if len(content) < 50 and not isinstance(element, TableItem):
                    continue
                
                chunks.append({
                    "chunk_id": f"{source_name}_c{chunk_counter:06d}", # FIXED
                    "source_file": source_name,
                    "page_number": actual_page, 
                    "section_path": current_section, 
                    "text": content,
                    "chunk_length": len(content),
                    "type": "table" if isinstance(element, TableItem) else "text",
                    "image_ref": None,
                    "image_description": None
                })
                chunk_counter += 1 # Increment our global tracker
            
            # Images
            elif isinstance(element, PictureItem) and element.image:
                try:
                    img = element.image.pil_image
                    
                    is_valid, img_hash = is_valid_image(img, seen_hashes)
                    if not is_valid: continue
                    seen_hashes.add(img_hash)
                    
                    img_name = f"asset_{counter:04d}.png"
                    img_path = assets_dir / img_name
                    
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    img.save(img_path)
                    
                    if vlm_enabled:
                        desc = vlm.describe(img_path, source_name)
                        if desc == "IGNORE_IMAGE": continue 
                    else:
                        desc = f"[Image: {img_name} from {source_name}, Page {actual_page}]"
                    
                    chunks.append({
                        "chunk_id": f"{source_name}_img_{chunk_counter:06d}", # FIXED
                        "source_file": source_name,
                        "page_number": actual_page,
                        "section_path": current_section,
                        "text": desc,
                        "chunk_length": len(desc),
                        "type": "visual_content",
                        "image_ref": str(img_path),
                        "image_description": desc
                    })
                    chunk_counter += 1 # Increment our global tracker
                    counter += 1 # Increment image filename counter
                    
                except Exception as img_e:
                    logger.debug(f"Image extraction skipped: {img_e}")
        
        gc.collect()
        # Return chunk_counter so the parent loop remembers it!
        return chunks, counter, chunk_counter
        
    except Exception as e:
        logger.error(f"Chunk failed: {e}")
        return [], counter, chunk_counter

def process_large_pdf(source: str, assets_dir: Path, counter: int, vlm_enabled: bool, seen_hashes: set):
    """Process PDF in chunks to avoid RAM crashes, routing to Docling."""
    source_name = os.path.basename(source)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        chunks_info = split_pdf_to_chunks(source, tmp_path)
        logger.info(f"Created {len(chunks_info)} chunks for Docling to process.")
        
        all_chunks = []
        chunk_counter = 0 # NEW: Initialize the tracker for this document
        
        for chunk_path, offset in chunks_info:
            logger.info(f"Processing pages {offset+1}-{offset+CHUNK_SIZE}...")
            
            # Pass chunk_counter in, and catch it when it comes back updated
            chunks, counter, chunk_counter = process_chunk(
                chunk_path, offset, source_name, assets_dir, counter, vlm_enabled, seen_hashes, chunk_counter
            )
            
            all_chunks.extend(chunks)
            os.remove(chunk_path)
        
        return all_chunks, counter


# ==================== Docling Routing & Web Scraper ====================
def ingestion_standard(source):
    """Standard Docling processor - used for DOCX/PPTX/HTML"""
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.DOCX: WordFormatOption(),
            InputFormat.PPTX: PowerpointFormatOption(),
            InputFormat.HTML: HTMLFormatOption()
        }
    )
    result = doc_converter.convert(source)
    return result.document

def download_web_images_concurrent(url: str, output_dir: Path, headers: dict, max_workers: int = 4) -> List[Dict]:
    """Scrape and download web images with parallel processing"""
    saved_images = []
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        image_urls = set()

        for img in soup.find_all('img'):
            src = img.get('src') or img.get('data-src')
            if src: image_urls.add(urljoin(url, src))

        filtered_urls = [u for u in image_urls if not any(x in u.lower() for x in [".svg", "logo", "icon"])]
        
        def download_single(img_url: str) -> Optional[Dict]:
            url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
            image_name = f"web_{url_hash}.png"
            save_path = output_dir / image_name
            if save_path.exists(): return {"name": image_name, "path": save_path, "type": "web"}
            try:
                img_data = requests.get(img_url, headers=headers, timeout=10)
                if img_data.status_code == 200:
                    img_obj = Image.open(BytesIO(img_data.content))
                    if img_obj.width > 150 and img_obj.height > 150:
                        img_obj.save(save_path, "PNG")
                        return {"name": image_name, "path": save_path, "type": "web"}
            except: pass
            return None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_single, url): url for url in filtered_urls}
            for future in tqdm(as_completed(futures), total=len(filtered_urls), desc="Web Images"):
                if result := future.result(): saved_images.append(result)
        return saved_images
    except Exception as e:
        logger.error(f"Web scrape error: {e}")
        return []

def process_single_source(source: str, assets_dir: Path, global_counter: int, 
                          vlm_enabled: bool = True, seen_hashes: set = None) -> tuple[List[Dict], int]:
    """Route inputs to the correct Docling ingestion method."""
    if seen_hashes is None:
        seen_hashes = set()
        
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # === 1. PDF Processing (Chunked Docling) ===
    if source.lower().endswith('.pdf') or (os.path.exists(source) and source.lower().endswith('.pdf')):
        return process_large_pdf(source, assets_dir, global_counter, vlm_enabled, seen_hashes)
    
    # === 2. Non-PDF Processing (Standard Docling) ===
    try:
        doc = ingestion_standard(source)
        source_name = os.path.basename(source) if os.path.exists(source) else source
        
        current_section = "General"
        current_page = 1
        chunks = []
        vlm = get_vlm() if vlm_enabled else None
        
        for element, level in doc.iterate_items():
            if hasattr(element, "prov") and element.prov:
                current_page = element.prov[0].page_no
            
            if isinstance(element, TextItem):
                if element.label == DocItemLabel.SECTION_HEADER:
                    current_section = element.text.strip()
                elif element.label == DocItemLabel.PAGE_HEADER:
                    continue
            
            if isinstance(element, (TextItem, TableItem)):
                content = element.export_to_markdown() if isinstance(element, TableItem) else str(element.text)
                content = re.sub(r'\s+', ' ', content).strip()
                
                if len(content) < 50 and not isinstance(element, TableItem):
                    continue
                
                chunks.append({
                    "chunk_id": f"{source_name}_c{len(chunks):06d}",
                    "source_file": source_name,
                    "page_number": int(current_page),
                    "section_path": current_section,
                    "text": content,
                    "chunk_length": len(content),
                    "type": "table" if isinstance(element, TableItem) else "text",
                    "image_ref": None,
                    "image_description": None
                })
                
            elif isinstance(element, PictureItem):
                if element.image and element.image.pil_image:
                    img = element.image.pil_image
                    
                    is_valid, img_hash = is_valid_image(img, seen_hashes)
                    if not is_valid: continue
                    seen_hashes.add(img_hash)
                    
                    image_name = f"asset_{global_counter}.png"
                    image_path = assets_dir / image_name
                    
                    if img.mode in ('RGBA', 'P'): img = img.convert('RGB')
                    img.save(image_path)
                    
                    if vlm_enabled:
                        desc = vlm.describe(image_path, source_name)
                        if desc == "IGNORE_IMAGE": continue
                    else:
                        desc = f"[Image: {image_name} from {source_name}]"
                    
                    chunks.append({
                        "chunk_id": f"{source_name}_img_{len(chunks):06d}",
                        "source_file": source_name,
                        "page_number": int(current_page),
                        "section_path": current_section,
                        "text": desc,
                        "chunk_length": len(desc),
                        "type": "visual_content",
                        "image_ref": str(image_path),
                        "image_description": desc
                    })
                    global_counter += 1
        
        # Concurrent web image downloading
        if source.startswith("http"):
            web_images = download_web_images_concurrent(source, assets_dir, headers)
            for img_meta in web_images:
                desc = vlm.describe(img_meta['path'], source_name) if vlm_enabled else f"[Web Image: {img_meta['name']}]"
                if desc != "IGNORE_IMAGE":
                    chunks.append({
                        "chunk_id": f"{source_name}_webimg_{len(chunks):06d}",
                        "source_file": source_name,
                        "page_number": 1,
                        "section_path": "Web Content",
                        "text": desc,
                        "chunk_length": len(desc),
                        "type": "visual_content",
                        "image_ref": str(img_meta['path']),
                        "image_description": desc
                    })
                    global_counter += 1
        
        logger.info(f"Generated {len(chunks)} chunks from {source_name}")
        return chunks, global_counter
        
    except Exception as e:
        logger.error(f"ERROR processing {source}: {e}")
        return [], global_counter

# ==================== Deduplication ====================
def deduplicate_sources(sources: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for source in sources:
        key = os.path.basename(source) if os.path.exists(source) else source.strip().rstrip('/')
        if key not in seen:
            deduped.append(source)
            seen.add(key)
        else:
            logger.warning(f"Removing duplicate from queue: {key}")
    return deduped

# ==================== Main Update Function ====================
def update_knowledge_base(new_sources: List[str], kb_path: str = "educational_knowledge_base.csv", 
                           assets_dir: str = "project_data/assets", vlm_enabled: bool = True) -> int:
    new_sources = deduplicate_sources(new_sources)
    logger.info(f"Processing {len(new_sources)} unique sources")
    
    assets_path = Path(assets_dir)
    assets_path.mkdir(parents=True, exist_ok=True)
    
    kb_writer = KnowledgeBaseWriter(kb_path)
    existing_sources = kb_writer.get_existing_sources()
    global_counter = kb_writer.get_visual_asset_count()
    
    total_new_chunks = 0
    
    for source in tqdm(new_sources, desc="Processing sources"):
        source_key = os.path.basename(source) if os.path.exists(source) else source
        
        if source_key in existing_sources:
            logger.info(f"Skipping {source_key}: already in KB")
            continue
            
        try:
            # We pass a fresh seen_hashes set per document to prevent cross-document logo collisions
            doc_seen_hashes = set()
            chunks, global_counter = process_single_source(source, assets_path, global_counter, vlm_enabled, doc_seen_hashes)
            
            if chunks:
                kb_writer.append_chunks(chunks)
                total_new_chunks += len(chunks)
                logger.info(f"Added {len(chunks)} chunks from {source_key}")
                existing_sources.add(source_key)
                
                if total_new_chunks % 1000 == 0:
                    gc.collect()
                    
        except Exception as e:
            logger.error(f"Failed to process {source}: {e}")
            continue
    
    logger.info(f"\nUpdate complete. Added {total_new_chunks} new chunks.")
    
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f) - 1
        return total_lines
    except:
        return total_new_chunks

if __name__ == "__main__":
    sources = []
    print("\n=== EduRAG Pro Knowledge Base Ingestion (100% Docling Chunked Edition) ===")
    print("Commands: Paste URL, Drag & Drop file path, or type 'done'\n")
    
    while True:
        try:
            user_input = input(">> Enter source: ").strip()
            if user_input.lower() == "done":
                if not sources:
                    print("No sources provided. Exiting.")
                    sys.exit(0)
                break
            
            clean_path = user_input.replace('"', '').replace("'", "")
            if clean_path.startswith("http") or Path(clean_path).exists():
                sources.append(clean_path)
                print(f"   Queued: {clean_path}")
            else:
                print(f"   Invalid path/URL: {clean_path}")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
    
    total = update_knowledge_base(
        sources, 
        kb_path="educational_knowledge_base.csv", 
        assets_dir="project_data/assets",
        vlm_enabled=True  # Set to False if you want to bypass the image descriptions for speed
    )
    print(f"\nTotal chunks in KB: {total}")
