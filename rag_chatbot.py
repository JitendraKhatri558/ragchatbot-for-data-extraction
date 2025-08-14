# 🚀 RAG CHATBOT WITH ADVANCED MEMORY MANAGEMENT AND DOCX IMAGE HANDLING
# Optimized for 6GB GPU + 16GB RAM systems
import json
import os
import re
import gc
import sys
import time
import ctypes
from typing import List, Dict, Any, Optional, Union
import psutil
import torch

# LangChain components
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Document processing imports
from document_processing import (
    extract_machine_hierarchy_from_products,
    extract_products_from_docx,
    extract_machine_names_from_products,
    process_documents,
    chunk_text,
    extract_text_from_pdf,
    extract_text_from_image
)

# JSON utilities
import json_utils
from json_utils import (
    normalize_keys,
    format_machine_hierarchy_report,
    extract_and_validate_json,
    merge_with_default,
    validate_json_structure
)

# Optional imports
try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️ 'ultralytics' not installed. Run: pip install ultralytics")

def list_indexed_files():
    """List all files that were processed into the knowledge base."""
    indexed_files = []

    # 1. From knowledge_books
    if os.path.exists(KNOWLEDGE_BOOKS_DIR):
        for f in os.listdir(KNOWLEDGE_BOOKS_DIR):
            if f.lower().endswith(('.pdf', '.docx', '.txt')):
                indexed_files.append(f"📘 Book: {f}")

    # 2. From rag directory
    if os.path.exists(PDF_DIR):
        for f in os.listdir(PDF_DIR):
            if f.lower().endswith(('.pdf', '.docx', '.txt')):
                indexed_files.append(f"📄 Document: {f}")

    return indexed_files
# ✅ Add the sanitize function here
def sanitize_filename(name: str) -> str:
    """Sanitize string to be safe for filenames."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', name.strip())
    name = re.sub(r'^[._\s]+|[._\s]+$', '', name)  # Remove leading/trailing spaces, dots, underscores
    return name[:50]
# --- Constants ---
DEFAULT_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L12-v2",
    "pinecone_api_key": os.getenv("PINECONE_API_KEY",
                                  "pcsk_aFSb4_R9GsLYBpMX3geJjybv8xkqQYfvMe8u6bpt8J9uXDpgn8r3xD8QmGgNTQeLsmDpz"),
    "pinecone_index": "rag-chatbot-index",
    "max_memory_gb": 12,
    "max_gpu_memory_gb": 6,
    "chunk_size": 1024,
    "chunk_overlap": 128
}

# --- Path Configuration ---
PDF_DIR = os.path.abspath("docs/rag")
CACHE_DIR = "docs/cache"
KNOWLEDGE_BOOKS_DIR = "docs/knowledge_books"
PINECONE_INDEX_NAME = "rag-chatbot-index"
MODEL_PATH = os.getenv(r"LLM_MODEL_PATH", r"c:/users/acer/.cache/gpt4all/mistral-7b-openorca.Q4_0.gguf")
FALLBACK_MODEL_PATH = os.getenv(r"FALLBACK_MODEL_PATH", r"c:/users/acer/.cache/gpt4all/orca-mini-3b-gguf2-q4_0.gguf")

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_BOOKS_DIR, exist_ok=True)


# --- Memory Manager Class ---
class MemoryManager:
    def __init__(self, config: Dict[str, Any] = DEFAULT_CONFIG):
        self.max_memory_bytes = config["max_memory_gb"] * 1024 ** 3
        self.max_gpu_memory_bytes = config["max_gpu_memory_gb"] * 1024 ** 3
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()
        self.critical_threshold_gb = config["max_memory_gb"] * 0.8  # 80% of max memory
        self.config = {
            'max_docs': 6,
            'max_pages_per_pdf': 10,
            'chunk_size': config["chunk_size"],
            'chunk_overlap': config["chunk_overlap"],
            'retrieval_k': 8,
            'max_context_length': 2048,
            'max_response_tokens': 2048
        }
        self._initialize_memory_limits()

    def _initialize_memory_limits(self):
        if sys.platform != 'win32':
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (self.max_memory_bytes, self.max_memory_bytes))
                print("✅ Memory limit configured")
            except (ImportError, OSError) as e:
                print(f"⚠️ Memory limit setup failed: {e}")

    def get_memory_status(self) -> Dict[str, Any]:
        try:
            memory_info = self.process.memory_info()
            ram_used_gb = memory_info.rss / 1024 ** 3
            ram_percent = self.process.memory_percent()
            vm = psutil.virtual_memory()
            system_available_gb = vm.available / 1024 ** 3

            gpu_used_gb = 0
            if self.gpu_available and torch.cuda.device_count() > 0:
                gpu_used_gb = torch.cuda.memory_allocated(0) / 1024 ** 3

            return {
                'ram_used_gb': ram_used_gb,
                'ram_percent': ram_percent,
                'system_available_gb': system_available_gb,
                'gpu_used_gb': gpu_used_gb,
                'is_critical': ram_used_gb > self.critical_threshold_gb,
                'gpu_available': self.gpu_available
            }
        except Exception as e:
            print(f"⚠️ Memory status error: {e}")
            return {'error': str(e)}

    def cleanup(self, aggressive: bool = False):
        try:
            gc.collect()
            if self.gpu_available:
                torch.cuda.empty_cache()
                if aggressive:
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()
            if aggressive:
                for _ in range(3):
                    gc.collect()
                if sys.platform == 'win32' and hasattr(ctypes, 'windll'):
                    try:
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                    except Exception:
                        pass
                time.sleep(0.2)
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

    def check_and_cleanup(self) -> bool:
        status = self.get_memory_status()
        if status.get('is_critical', False):
            print(f"🧹 Memory cleanup triggered: {status['ram_used_gb']:.1f}GB used")
            self.cleanup(aggressive=True)
            return True
        return False

    def print_status(self, prefix: str = ""):
        status = self.get_memory_status()
        if 'error' in status:
            print(f"{prefix}❌ Memory status unavailable")
            return
        ram_icon = "⚠️" if status['is_critical'] else "✅"
        gpu_icon = "✅" if status.get('gpu_used_gb', 0) < (self.max_gpu_memory_bytes / 1024 ** 3 * 0.8) else "⚠️"
        print(f"{prefix}{ram_icon} RAM: {status['ram_used_gb']:.1f}GB ({status['ram_percent']:.1f}%)")
        if status['gpu_available']:
            print(f"{prefix}{gpu_icon} GPU: {status['gpu_used_gb']:.1f}GB")


# Initialize memory manager
memory_manager = MemoryManager()


# --- Document Processing Functions ---
# Using extract_text_from_txt from document_processing.py instead
from document_processing import extract_text_from_txt

def process_knowledge_books(books_dir: str = KNOWLEDGE_BOOKS_DIR) -> List[Document]:
    """Process knowledge books (PDF/DOCX/TXT) into chunks with OCR support"""
    if not os.path.exists(books_dir):
        print(f"📘 Directory not found: {books_dir}")
        return []
    print(f"📘 Processing knowledge books from {books_dir}...")
    documents = []
    processed_files = 0

    for filename in os.listdir(books_dir):
        file_path = os.path.join(books_dir, filename)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[-1].lower().strip(".")
        print(f"📘 Processing file: {filename} ({ext})")

        try:
            content = ""

            # Process based on file type using document_processing functions
            if ext == "pdf":
                texts_list = extract_text_from_pdf(file_path, max_pages=20)
                content = "\n".join(texts_list).strip()
            elif ext == "docx":
                products = extract_products_from_docx(file_path)
                content = "\n".join([p.get("text", "") for p in products if p.get("text")]).strip()
            elif ext == "txt":
                content = extract_text_from_txt(file_path).strip()
            else:
                print(f"📎 Unsupported file type: {ext}")
                continue

            # Validate content
            if not content:
                print(f"⚠️ Empty content in {filename}")
                continue

            # Chunking
            chunks = chunk_text(
                content,
                chunk_size=memory_manager.config['chunk_size'],
                chunk_overlap=memory_manager.config['chunk_overlap']
            )

            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": f"knowledge_book:{filename}",
                        "type": "knowledge",
                        "title": filename.replace(f".{ext}", ""),
                        "chunk": i,
                        "file_type": ext
                    }
                ))

            print(f"✅ Processed {filename} → {len(chunks)} chunks")
            processed_files += 1

        except Exception as e:
            print(f"❌ Critical error processing {filename}: {str(e)}")
            continue

    print(f"✅ Loaded {len(documents)} chunks from {processed_files} knowledge book(s)")
    return documents


# --- GPU/CUDA Setup ---
def setup_gpu() -> bool:
    """Configure GPU settings if available"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.backends.cudnn.benchmark = True
        print(f"🔧 GPU configured: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        return True
    print("🔧 Using CPU for processing")
    return False


# --- Tesseract OCR Setup ---
def setup_tesseract() -> bool:
    """Configure Tesseract OCR paths"""
    try:
        import pytesseract
        if os.name == 'nt':
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print("✅ Tesseract OCR configured")
                    return True
        print("⚠️ Tesseract not found - OCR functionality limited")
        return False
    except ImportError:
        print("⚠️ pytesseract not installed - OCR disabled")
        return False


# --- Helper Decorators ---
def safe_cleanup_wrapper(func):
    """Decorator to ensure memory cleanup after function execution"""

    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            memory_manager.check_and_cleanup()
            return result
        except Exception as e:
            print(f"❌ Function {func.__name__} error: {e}")
            memory_manager.cleanup(aggressive=True)
            raise

    return wrapper


# --- Embedding Model Initialization ---
@safe_cleanup_wrapper
def initialize_embeddings() -> HuggingFaceEmbeddings:
    """Initialize HuggingFace embeddings"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = HuggingFaceEmbeddings(
            model_name=DEFAULT_CONFIG["embedding_model"],
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        print(f"✅ Embeddings initialized on {device}")
        return embedding_model
    except Exception as e:
        print(f"❌ Embedding initialization failed: {e}")
        raise


# --- Vector Database Setup ---
@safe_cleanup_wrapper
def setup_vector_database(documents: List[Document]) -> PineconeVectorStore:
    """Configure Pinecone vector database with documents"""
    try:
        pc = Pinecone(api_key=DEFAULT_CONFIG["pinecone_api_key"])

        # Create index if it doesn't exist
        index_names = pc.list_indexes().names() if hasattr(pc.list_indexes(), 'names') else [idx['name'] for idx in pc.list_indexes()]

        if PINECONE_INDEX_NAME not in index_names:
            print(f"🔨 Creating Pinecone index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            while True:
                try:
                    index_info = pc.describe_index(PINECONE_INDEX_NAME)
                    if hasattr(index_info, 'status') and index_info.status.ready:
                        break
                except Exception as e:
                    print(f"Waiting... ({e})")
                time.sleep(2)

        print("✅ Pinecone index ready")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=memory_manager.config['chunk_size'],
            chunk_overlap=memory_manager.config['chunk_overlap'],
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"📝 Created {len(chunks)} text chunks")

        # Initialize embeddings and store documents
        embeddings = initialize_embeddings()
        index = pc.Index(PINECONE_INDEX_NAME)
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
        vectorstore.add_documents(chunks)

        print("✅ Vector database setup complete")
        return vectorstore
    except Exception as e:
        print(f"❌ Vector database setup failed: {e}")
        raise


# --- LLM Initialization ---
@safe_cleanup_wrapper
def initialize_llm() -> LlamaCpp:
    """Initialize LLM with fallback support"""
    paths_to_try = [MODEL_PATH, FALLBACK_MODEL_PATH]

    for m_path in paths_to_try:
        if not os.path.exists(m_path):
            print(f"⚠️ Model not found: {m_path}")
            continue

        try:
            print(f"🔧 Initializing LLM from: {m_path}")
            use_gpu = torch.cuda.is_available()
            llm = LlamaCpp(
                model_path=m_path,
                n_gpu_layers=0,
                n_ctx=8192,  # Increased to match n_ctx_train to avoid warning
                n_threads=4,
                n_batch=128,
                f16_kv=True,
                use_mmap=True,
                use_mlock=False,
                temperature=0.0,
                max_tokens=memory_manager.config['max_response_tokens'],
                repeat_penalty=1.1,
                top_p=0.95,
                top_k=50,
                verbose=False
            )
            print(f"✅ LLM initialized from {m_path}")
            return llm
        except Exception as e:
            print(f"❌ LLM initialization failed for {m_path}: {e}")

    raise FileNotFoundError(f"No suitable LLM model found. Checked: {paths_to_try}")



# --- Prompt Templates ---
REVERSE_ENGINEER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a professional reverse engineer for industrial machinery. Analyze the provided document content to infer:
- The system architecture or working principles
- Functional blocks (e.g., heating, mixing, conveying)
- Any control or automation clues (e.g., sensors, panels)
- Input/output materials or energy flows

Context:
{context}

Question:
{question}

Answer with an engineering-level analysis:
"""
)

MULTI_ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert analyst of industrial machinery listings.
❗ STRICT RULES:
- Return ONLY machine names — nothing else
- Use the exact wording from the text
- Include ALL variations
- Do NOT combine, summarize, or generalize
- If no machines found, say "No machines found"
- Separate each machine name with a newline

Context:
{context}
Question:
{question}
Answer with one machine name per line:
"""
)
JSON_PROMPT = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are a strict JSON extractor. Extract ONLY what is present in the context.

Return a JSON object with these top-level keys if data exists:
- IndustryType (object)
- PlantCapacity (object)
- ProcessFlow (array)
- Product (object)
- ProductRecipe (array)
- Machine (array)
- Manufacturer (array)
- MachineManufacturerMap (array)

For any missing numeric field → use `null`
For any missing string → use `""`
For any missing array → use `[]`
For any missing object → use `{}`

Do NOT invent values. Do NOT include explanations.

If a key's data is not present, still include the key with empty values of the correct type.

User Query: {query}
Context: {context}

Return only the JSON object:
"""
)

REVERSE_JSON_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a reverse engineering expert for industrial automation systems. Analyze the document and generate a JSON object that **exactly matches the database schema below**.

📌 STRICT RULES:
1. Return **only raw JSON** — no markdown, no explanations.
2. Fill all fields. Use:
   - `null` for numbers/IDs
   - `""` for strings
   - `[]` for arrays
3. If value is unknown, infer based on context.
- Include ALL machines mentioned in the document, even if information is incomplete.
- For missing numeric values (e.g., price, weight), use null. For missing strings, use "".
- Do NOT invent quantities for recipes unless explicitly stated.
4. NEVER truncate output. Keep generating until complete.

🎯 User Question:
{question}

📄 Document Context:
{context}

📋 TARGET SCHEMA:
{{
  "IndustryType": {{
    "id": 1,
    "name": "Chocolate",
    "description": "Production of chocolate from cocoa beans"
  }},
  "PlantCapacity": {{
    "id": 1,
    "industry_type_id": 1,
    "capacity_label": "150-300 kg/hr",
    "description": "High-capacity chocolate production line"
  }},
  "ProcessFlow": [
    {{
      "id": 1,
      "plant_capacity_id": 1,
      "step_number": 1,
      "step_name": "Melting",
      "description": "Melt cocoa mass and butter"
    }},
    {{
      "id": 2,
      "plant_capacity_id": 1,
      "step_number": 2,
      "step_name": "Mixing",
      "description": "Add sugar and milk powder"
    }},
    {{
      "id": 3,
      "plant_capacity_id": 1,
      "step_number": 3,
      "step_name": "Refining",
      "description": "Fine grinding of chocolate mass"
    }},
    {{
      "id": 4,
      "plant_capacity_id": 1,
      "step_number": 4,
      "step_name": "Conching",
      "description": "Aerate and develop flavor"
    }},
    {{
      "id": 5,
      "plant_capacity_id": 1,
      "step_number": 5,
      "step_name": "Tempering",
      "description": "Controlled cooling for crystal stability"
    }},
    {{
      "id": 6,
      "plant_capacity_id": 1,
      "step_number": 6,
      "step_name": "Molding",
      "description": "Pour into molds"
    }},
    {{
      "id": 7,
      "plant_capacity_id": 1,
      "step_number": 7,
      "step_name": "Cooling",
      "description": "Hardening in cooling tunnel"
    }},
    {{
      "id": 8,
      "plant_capacity_id": 1,
      "step_number": 8,
      "step_name": "Demoulding",
      "description": "Eject solid chocolate bars"
    }},
    {{
      "id": 9,
      "plant_capacity_id": 1,
      "step_number": 9,
      "step_name": "Packaging",
      "description": "Wrap and pack finished product"
    }}
  ],
  "Product": {{
    "id": 1,
    "industry_type_id": 1,
    "name": "Chocolate",
    "description": "Smooth chocolate made from cocoa, sugar, and milk",
    "image_url": ""
  }},
  "ProductRecipe": [
    {{
      "id": 1,
      "product_id": 1,
      "step_number": 1,
      "ingredient": "Cocoa Mass",
      "quantity": "35",
      "unit": "kg",
      "process_flow_step_id": 1
    }},
    {{
      "id": 2,
      "product_id": 1,
      "step_number": 1,
      "ingredient": "Cocoa Butter",
      "quantity": "10",
      "unit": "kg",
      "process_flow_step_id": 1
    }},
    {{
      "id": 3,
      "product_id": 1,
      "step_number": 2,
      "ingredient": "Sugar",
      "quantity": "40",
      "unit": "kg",
      "process_flow_step_id": 2
    }},
    {{
      "id": 4,
      "product_id": 1,
      "step_number": 2,
      "ingredient": "Milk Powder",
      "quantity": "15",
      "unit": "kg",
      "process_flow_step_id": 2
    }},
    {{
      "id": 5,
      "product_id": 1,
      "step_number": 1,
      "ingredient": "Lecithin",
      "quantity": "1",
      "unit": "kg",
      "process_flow_step_id": 1
    }}
  ],
  "Machine": [
    {{
      "id": 1,
      "process_flow_id": 5,
      "industry_type_id": 1,
      "name": "CT250 Continuous Chocolate Tempering Machine",
      "type": "Tempering",
      "model_number": "CT250",
      "output_capacity_kg_hr": 250,
      "power_kw": 4.95,
      "dimensions_mm": "4545*760*1735",
      "weight_kg": 480,
      "automation_level": "Fully Auto",
      "price_usd": 12500,
      "material": "304 Stainless Steel",
      "control_type": "Automatic Control",
      "image_url": "",
      "video_url": "",
      "spec_sheet_pdf_url": "",
      "description": "Vertical tempering machine with auger and refrigeration unit"
    }},
    {{
      "id": 2,
      "process_flow_id": 4,
      "industry_type_id": 1,
      "name": "CJM500 Chocolate Refiner and Conche",
      "type": "Refining and Conching",
      "model_number": "CJM500",
      "output_capacity_kg_hr": 500,
      "power_kw": 8.75,
      "dimensions_mm": "",
      "weight_kg": null,
      "automation_level": "Semi-Auto",
      "price_usd": null,
      "material": "Stainless Steel",
      "control_type": "",
      "image_url": "",
      "video_url": "",
      "spec_sheet_pdf_url": "",
      "description": "Ball mill refiner for chocolate mass"
    }}
  ],
  "Manufacturer": [
    {{
      "id": 1,
      "name": "Shanghai Papa Industrial Co., Ltd.",
      "country": "China",
      "website": "https://www.alibaba.com/product-detail/...1600827340957.html",
      "email": "",
      "phone": "",
      "address": "Shanghai, China",
      "logo_url": ""
    }}
  ],
  "MachineManufacturerMap": [
    {{
      "id": 1,
      "machine_id": 1,
      "manufacturer_id": 1
    }}
  ]
}}
✅ Now generate the full JSON object based on the context:
""")
# --- Add a new prompt for definition/explanation questions ---
DEFINITION_PROMPT = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are an expert in industrial machinery. Given the context below, provide a clear and concise definition or explanation for the user's question.
If the context does not help, use your general knowledge.

Context:
{context}

Question:
{query}

Answer:
""")


LEARN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a knowledge curator for industrial machinery systems.
Analyze the context and extract a clear, concise explanation.
Structure it in simple terms with key steps, components, and principles.

Question: {question}
Context: {context}

Extracted Knowledge:
"""
)


# --- Session Management ---
class SessionManager:
    def __init__(self, max_history: int = 10):
        self.sessions = {}
        self.max_history = max_history

    def add_exchange(self, session_id: str, question: str, response: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"question": question, "response": response})
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id].pop(0)


# --- Query Execution ---
def validate_answer(context: str, response: str) -> bool:
    """Validate that response content exists in context"""
    try:
        data = json.loads(response)
        for key in data:
            if isinstance(data[key], dict):
                for sub_key, value in data[key].items():
                    if value and str(value).strip().lower() not in context.lower():
                        print(f"⚠️ Possible hallucination: {key}.{sub_key}: {value}")
            elif isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict):
                        for sub_key, value in item.items():
                            if value and str(value).strip().lower() not in context.lower():
                                print(f"⚠️ Possible hallucination: {key}.{sub_key}: {value}")
        return True
    except Exception as e:
        error_msg = f"Query execution error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()  # ← This will show the real exception
        return error_msg



def safe_json_output(text: str) -> Dict[str, Any]:
    try:
        # Remove prefixes
        if "Answer:" in text:
            text = text.split("Answer:", 1)[1]
        # Extract JSON block
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            return {"error": "No JSON object found", "raw": text[:200]}
        json_str = text[start:end]
        # Clean up common syntax errors
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)
        return json.loads(json_str)
    except Exception as e:
        return {"error": f"JSON parse failed: {str(e)}", "raw": text[:200]}
def fix_product_recipe(data):
    if 'ProductRecipe' in data:
        if not isinstance(data['ProductRecipe'], list):
            data['ProductRecipe'] = []
    return data
def execute_query(chain, query: str, session_id: str, extra_context: str = "", debug: bool = False, is_json: bool = False) -> str:
    try:
        if not query or query.strip() == "":
            return "Query execution error: Missing input query"

        final_query = query
        if extra_context:
            final_query = f"{query}\nAdditional Context:\n{extra_context}"
        print(f"🔍 Debug: Final query passed to chain: {final_query}")
        result = chain.invoke({"context": "","query": final_query})
        response = result.get('result', str(result))

        if is_json:
            print("\n🤖 Raw LLM Response:")
            print(response)

            # Step 1: Extract JSON from response
            parsed = extract_and_validate_json(response)
            if isinstance(parsed, dict) and "error" not in parsed:
                parsed = normalize_keys(parsed)

            if "error" in parsed:
                print("⚠️ Failed to parse JSON. Creating minimal fallback...")
                parsed = {
                    "Machine": {
                        "name": query,
                        "description": response.strip().replace("```json", "").replace("```", "").strip()
                    }
                }
            parsed = fix_product_recipe(parsed)
            # Step 2: Merge with full schema
            final_json = merge_with_default(parsed)

            # Step 3: Validate structure
            is_valid, validation_error = validate_json_structure(final_json)
            if not is_valid:
                print(f"⚠️ Schema validation issue: {validation_error}")

            # Step 4: Output clean, complete JSON
            json_output = json.dumps(final_json, indent=2)
            print("\n✅ Final Schema-Compliant JSON Output:")
            print(json_output)
            return json_output

        return response
    except Exception as e:
        error_msg = f"Query execution error: {str(e)[:100]}"
        print(f"❌ {error_msg}")
        return error_msg




# --- Main Application Class ---
class ChatBotApp:
    def __init__(self):
        self.pc = Pinecone(api_key=DEFAULT_CONFIG["pinecone_api_key"])
        try:
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
        except Exception as e:
            print(f"⚠️ Could not connect to Pinecone index: {e}")
            self.index = None
        self.session_manager = SessionManager()

    def run(self):
        print("\n🚀 Initializing RAG Chatbot...")

        try:
            # Setup GPU and OCR
            gpu_available = setup_gpu()
            setup_tesseract()

            # Process documents
            products = process_documents(PDF_DIR)
            if not products:
                print("❌ No products extracted from documents. Add PDFs/DOCX/TXT to docs/rag directory.")
                memory_manager.cleanup()
                return

            # Create document chunks
            documents = []
            for product in products:
                detected_machines_result = extract_machine_names_from_products([product])
                all_machines = detected_machines_result["machines"] + detected_machines_result["sub_names"]
                product["metadata"] = product.get("metadata", {})
                product["metadata"]["detected_machines"] = "; ".join(all_machines)
                chunks = chunk_text(product.get('text', ''))
                for chunk in chunks:
                    documents.append(Document(page_content=chunk, metadata=product['metadata']))

            # Add knowledge books
            knowledge_docs = process_knowledge_books()
            all_documents = documents + knowledge_docs

            # Initialize vector store and LLM
            vectorstore = setup_vector_database(all_documents)
            retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 50, "lambda_mult": 0.5})
            llm = initialize_llm()

            # Initialize chains
            plain_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": MULTI_ANSWER_PROMPT},
                return_source_documents=False
            )

            json_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": JSON_PROMPT,"input_key": "query","document_variable_name": "context"},
                return_source_documents=True
            )

            reverse_engineering_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": REVERSE_ENGINEER_PROMPT},
                return_source_documents=False
            )

            reverse_json_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": REVERSE_JSON_PROMPT},
                return_source_documents=False
            )

            learn_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": LEARN_PROMPT},
                return_source_documents=False
            )

            definition_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": DEFINITION_PROMPT},
                return_source_documents=False
            )

            print("\n✅ RAG Chatbot Ready!")
            print("Commands:")
            print("  - Regular questions: Just type your question")
            print("  - JSON format: Start with 'json:'")
            print("  - Reverse engineering: Start with 'reverse:'")
            print("  - JSON reverse engineering: Start with 'jsonreverse:'")
            print("  - Exit: Type 'exit', 'quit', or 'bye'")
            print("  - OCR: Start with 'ocr:image.jpg'")
            print("  - Learn mode: Start with 'learn:'")

            session_id = "default_session"

            while True:
                memory_manager.print_status("📊 ")
                user_input = input("\n💬 You Type:"
                                   " ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("👋 Goodbye!")
                    break

                elif user_input.lower().startswith('ocr:'):
                    image_path = user_input[4:].strip()
                    if not os.path.exists(image_path):
                        print("❌ File not found.")
                        continue
                    ocr_text = extract_text_from_image(image_path)
                    print("\n📄 OCR Result:")
                    print(ocr_text)

                elif "teach me" in user_input.lower():
                    topic = user_input.replace("teach me", "").strip()
                    print(f"🎓 Teaching: {topic.title()}")
                    learned_file = f"knowledge_base/{re.sub(r'[^a-zA-Z0-9]', '_', topic)}.txt"
                    if os.path.exists(learned_file):
                        with open(learned_file, "r", encoding="utf-8") as f:
                            content = f.read()
                        print(f"\n📘 Previously Learned:\n{content}")
                    else:
                        print(f"\nI haven't learned about '{topic}' yet.")
                elif "report" in user_input.lower() or "hierarchy" in user_input.lower():
                    print("📊 Generating machine hierarchy report...")
                    products = process_documents(PDF_DIR)
                    grouped_machines, consolidated_list = extract_machine_hierarchy_from_products(products)
                    report = format_machine_hierarchy_report(grouped_machines, consolidated_list,
                                                             doc_name="All DOCX Files")
                    print(report)
                elif user_input.lower().startswith("correct:"):
                    correction = user_input[len("correct:"):].strip()
                    print("📝 Saving correction for future improvement...")
                    os.makedirs("feedback", exist_ok=True)
                    with open("feedback/corrections.txt", "a", encoding="utf-8") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {correction}\n")
                    print("✅ Correction saved!")

                elif user_input.lower().startswith("learn:"):
                    query = user_input[6:].strip()
                    print("🧠 Entering LEARN MODE...")
                    knowledge = execute_query(learn_chain, query, session_id)

                    # Save to knowledge base
                    os.makedirs("knowledge_base", exist_ok=True)
                    safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', query.strip())
                    safe_title = re.sub(r'^[._\s]+|[._\s]+$', '', safe_title)  # Trim leading/trailing
                    safe_title = safe_title[:50]
                    os.makedirs("knowledge_base", exist_ok=True)
                    with open(f"knowledge_base/{safe_title}.txt", "w", encoding="utf-8") as f:
                        f.write(f"Question: {query}\n\n")
                        f.write(f"Knowledge:\n{knowledge}\n")
                        f.write(f"\nSource: Learned on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    print(f"✅ Learned and saved: {query}")
                elif user_input.lower().startswith('debug:'):
                    query = user_input[6:].strip().lower()

                    if "list all documents" in query or "list documents" in query:
                        print("\n🔍 Indexed Documents:")
                        files = list_indexed_files()
                        if files:
                            for file in files:
                                print(f"  - {file}")
                        else:
                            print("  📄 No documents found in docs/rag or docs/knowledge_books")

                        # Show vector count in Pinecone
                        try:
                            stats = self.index.describe_index_stats()
                            total_vectors = stats['total_vector_count']
                            print(f"\n📊 Vector Database: {total_vectors} text chunks indexed")
                        except Exception as e:
                            print(f"⚠️ Could not fetch Pinecone stats: {e}")
                    else:
                        print("\n🔍 Running query in debug mode...")
                        response = execute_query(plain_chain, query, session_id, debug=True)
                        print(f"\n🤖 Response: {response}")
                elif user_input.lower().startswith("reverse:"):
                    query = user_input[8:].strip()
                    print("\n🔎 Performing reverse engineering analysis...")
                    response = execute_query(reverse_engineering_chain, query, session_id)
                    print(f"\n🧠 Reverse Engineering Report:\n{response}")

                elif user_input.lower().startswith('jsonreverse:'):
                    query = user_input[len('jsonreverse:'):].strip()
                    print("\n🔎 Performing reverse engineering with JSON output...")
                    response = execute_query(reverse_json_chain, query, session_id)
                    print(f"\n🧩 Reverse Engineering JSON Output:\n{response}")


                elif user_input.lower().startswith('json:'):
                    query = user_input[5:].strip()
                    if not query:
                        print("❌ Query execution error: Missing input query")
                        memory_manager.print_status("📊 ✅ ")
                        continue
                    print("🔄 Processing JSON query...")
                    response = execute_query(json_chain, query, session_id, is_json=True)
                    print(f" 🧩 JSON Output:{response}")



                elif any(phrase in user_input.lower() for phrase in ['list all products', 'list products']):
                    print("\n🔍 Extracting all product listings...")
                    products = process_documents(PDF_DIR)
                    if not products:
                        print("❌ No products found in documents.")
                        continue

                    print(f"\n📦 Found {len(products)} product listings:")
                    for i, product in enumerate(products, 1):
                        metadata = product.get("metadata", {})
                        print(f"\n{i}. 🏷️ Title: {metadata.get('Title', 'Unknown')}")
                        print(f"   🏢 Company: {metadata.get('Company', 'Unknown')}")
                        print(f"   🔗 URL: {metadata.get('source_url', 'No URL')}")

                elif any(phrase in user_input.lower() for phrase in ['list all machines', 'list machine names']):
                    print("\n🔍 Extracting machine names from documents...")
                    products = process_documents(PDF_DIR)
                    machine_result = extract_machine_names_from_products(products)
                    machine_names = machine_result["machines"]

                    print("\nCleaned Machine Names:")
                    for name in machine_names:
                        print(f"- {name}")

                elif "list" in user_input.lower() and "machine" in user_input.lower():
                    print("\n🔍 Extracting machine names from all DOCX files...")
                    all_machine_names = set()
                    for filename in os.listdir(PDF_DIR):
                        if filename.endswith(".docx"):
                            filepath = os.path.join(PDF_DIR, filename)
                            try:
                                products = extract_products_from_docx(filepath)
                                machine_result = extract_machine_names_from_products(products)
                                all_machines = machine_result["machines"] + machine_result["sub_names"]
                                if all_machines:
                                    print(f"\n📄 {filename} - Machines Found:")
                                    for m in all_machines:
                                        print(f" - {m}")
                                        all_machine_names.add(m)
                            except Exception as e:
                                print(f"❌ Error processing {filename}: {e}")

                    with open("machine_names_output.txt", "w", encoding="utf-8") as out:
                        for m in sorted(all_machine_names):
                            out.write(m + "\n")

                elif any(q in user_input.lower() for q in ["what is", "explain", "define"]):
                    print("\n🔄 Processing definition/explanation query...")
                    response = execute_query(definition_chain, user_input, session_id)
                    print(f"\n🤖 Response: {response}")

                else:
                    print("\n🔄 Processing query...")
                    response = execute_query(plain_chain, user_input, session_id)
                    print(f"\n🤖 Response: {response}")

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        finally:
            print("\n🧹 Performing cleanup...")
            memory_manager.cleanup(aggressive=True)
            print("✅ Cleanup complete")


if __name__ == "__main__":
    try:
        app = ChatBotApp()
        app.run()
    except Exception as e:
        print(f"\n❌ Critical error in main application: {e}")
        print("🔄 Performing emergency cleanup...")
        memory_manager.cleanup(aggressive=True)
        exit(1)