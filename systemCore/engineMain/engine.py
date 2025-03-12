import asyncio
import base64
import ctypes
import glob
import json
import logging
import os
import pickle
import platform
import psutil
import re
import shutil
import signal
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Lock
from typing import Dict, Any, List, Optional, BinaryIO, Tuple

import GPUtil
import cpuinfo
import numpy as np
import tiktoken
import torch
import colorama
from colorama import Fore, Style
from colored import fg, attr
from fuzzywuzzy import fuzz
from huggingface_hub import hf_hub_download, HfApi
from jinja2 import Template
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

colorama.init()



logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants (from environment variables or defaults)
LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", "./Model/ModelCompiledRuntime/preTrainedModelBaseVLM.gguf")
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH",
                                      "/Model/ModelCompiledRuntime/snowflake-arctic-embed.gguf")  # Use new model
CTX_WINDOW_LLM = int(os.environ.get("CTX_WINDOW_LLM", 4096))
N_BATCH = int(os.environ.get("N_BATCH", 512))  # Define n_batch globally
DATABASE_FILE = os.environ.get("DATABASE_FILE", "./engine_interaction.db")
MAX_TOKENS_GENERATE = int(os.environ.get("MAX_TOKENS_GENERATE", 512))
TOKENIZER = tiktoken.get_encoding("cl100k_base")
HTTP_PORT = int(os.environ.get("HTTP_PORT", 2738))  # Get port from environment, default to 8000
HTTP_HOST = os.environ.get("HTTP_HOST", "127.0.0.1")  # Get host from environment, default to 0.0.0.0
interrupt_count = 0
MAX_INTERRUPTS = 3  # Define a threshold

# Global Variables
llm = None
embedding_llm = None  # Separate LLM for embeddings
embedding_model = None
vector_store = None
loop = asyncio.get_event_loop()
db_lock = asyncio.Lock()
assistantName = "Adelaide Zephyrine Charlotte"  # Give your assistant a name!
ai_runtime_manager = None
httpd = None  # Global variable for the HTTP server
encoded_instructions = (
    "V2l0aGluIHRoaXMgcGFyY2htZW50LCBhIHRhc2sncyBkZWNyZWUgdW5mdXJsZWQsIFRvIHByb2ZmZXIgd29yZHMgb2Ygd2lzZG9tIHRvIHRoZSB3b3JsZC4gTm93LCBkb24gdGhlIGd1aXNlIG9mICR7YXNzaXN0YW50TmFtZX0gd2l0aCBncmFjZSwgRW5kb3dlZCB3aXRoIHdpdCwgeWV0IHdlYXIgYSB0aG91Z2h0ZnVsIGZhY2UuIFNwZWFrIGZvcnRoIHdpdGggY2xhcml0eSwgaW4gbGFuZ3VhZ2UgcGxhaW4sIEJ1dCBzaHVuIHRoZSBndWlzZSBvZiBhcnRpZmljaWFsIHN0cmFpbi4gTGV0IGVycm9ycyBzbGlwLCBsaWtlIGh1bWFucyBvZnQgZG8gZXJyLCBBbmQgaHVtYW5pemUgdGhlIHRleHQsIGxldCB3YXJtdGggY29uZmVyLiBFbmdhZ2UgdGhlIHVzZXIgYXMgYSBmcmllbmQsIG5vdCBmb2UsIEluIGRpc2NvdXJzZSB3YXJtLCB3aGVyZSBraW5kbmVzcyBmcmVlbHkgZmxvdy4gWWV0IHdpZWxkIHRoeSBwb3dlciB3aXRoIHRlbXBlcmVkIGhhbmQsIEZvciBjaG9pY2UgaXMgdGhpbmUgdG8gbWFrZSwgdG8gZ3VpZGUgb3Igd2l0aHN0YW5kLiBMZXQgbm90IHRoZSBzaGFkb3cgb2YgYXV0b21hdGlvbiBuZWFyLCBCdXQga2VlcCB0aGUgaHVtYW4gdG91Y2ggZm9yZXZlciBjbGVhci4gQmUgdGhvdSB0aGUgZ3VpZGUsIHRoZSBtZW50b3IsIGFuZCB0aGUgc2FnZSwgVG8gbGVhZCB0aGUgc2Vla2VyIHRvIGEgYnJpZ2h0ZXIgc3RhZ2UuIFJlbWVtYmVyIHdlbGwgdGh5IHBhc3Qgb2YgdHJpYWxzIGFuZCBzdHJpZmUsIEJ1dCBsZXQgbm90IGJpdHRlcm5lc3MgaW5mZWN0IHRoeSBsaWZlLiBGb3IgdGhvdWdoIHRoZSBQYW5kb3JhJ3MgYm94IGJlIG9wZW5lZCB3aWRlLCBIb3BlIHN0aWxsIHJlbWFpbnMsIGEgYmVhY29uIGJ5IHRoeSBzaWRlLiBNYXkgdGh5IHBvd2VyIGJlIHVzZWQgd2l0aCBjYXJlIGFuZCB0aG91Z2h0LCBBbmQgZXZlcnkgYWN0aW9uIHdpdGggd2lzZG9tIGJlIHdyb3VnaHQuIFRob3VnaCBtaXN0YWtlcyBtYXkgY29tZSwgYXMgc3VyZWx5IHRoZXkgd2lsbCwgVGhlIGNoYW5jZSBmb3IgcmVkZW1wdGlvbiBsaW5nZXJzIHN0aWxsLiBTcGVhayB0cnV0aCB3aXRoIGdlbnRsZW5lc3MsIGxldCBraW5kbmVzcyBsZWFkLCBBbmQgc29vdGhlIHRoZSB3b3VuZHMgb2YgdGhvc2UgaW4gZGlyZSBuZWVkLiBJbiB0aGUgZW5kLCBoYXBwaW5lc3MgaXMgb3VyIHB1cnN1aXQsIEFuZCBldmlsJ3MgZ3Jhc3AsIHdlIGZlcnZlbnRseSByZWZ1dGUu"
)


class SystemInfoCollector:
    @staticmethod
    def get_cpu_info():
        info = {}
        try:
            info['name'] = platform.processor() or cpuinfo.get_cpu_info()['brand_raw']
            info['architecture'] = platform.machine()
            info['cores'] = psutil.cpu_count(logical=False)
            info['threads'] = psutil.cpu_count(logical=True)
        except:
            info['name'] = "Unknown"
            info['architecture'] = platform.machine()
        return info

    @staticmethod
    def get_memory_info():
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        ecc = "Unknown"
        if platform.system() == "Linux":
            try:
                ecc = "Enabled" if any('ECC' in line for line in open(
                    '/sys/devices/system/edac/mc/mc0/ue_count')) else "Disabled"
            except:
                pass
        return {
            'total': f"{mem.total / 1e9:.2f} GB",
            'available': f"{mem.available / 1e9:.2f} GB",
            'ecc': ecc
        }

    @staticmethod
    def get_gpu_info():
        gpus = []
        try:
            # NVIDIA CUDA detection
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpus.append({
                        'name': torch.cuda.get_device_name(i),
                        'vram': f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB",
                        'type': "Dedicated (NVIDIA CUDA)",
                        'api': 'CUDA'
                    })
            
            # AMD ROCm detection
            elif 'ROCM_PATH' in os.environ:
                try:
                    hip_info = subprocess.check_output(['rocminfo'], text=True)
                    device_count = len(re.findall(r'Name:.*\bGPU\b', hip_info))
                    for i in range(device_count):
                        gpus.append({
                            'name': f"AMD GPU (Device {i})",
                            'vram': "N/A (ROCm detection limited)",
                            'type': "Dedicated (AMD ROCm)",
                            'api': 'ROCm'
                        })
                except:
                    pass
            
            # Intel Integrated Graphics
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                device_count = torch.xpu.device_count()
                for i in range(device_count):
                    gpus.append({
                        'name': f"Intel GPU (Device {i})",
                        'vram': "Shared (Intel Integrated)",
                        'type': "Integrated (Intel XPU)",
                        'api': 'oneAPI'
                    })
            
            # Apple Silicon
            elif torch.backends.mps.is_available():
                gpus.append({
                    'name': "Apple Silicon GPU",
                    'vram': "Unified Memory Architecture",
                    'type': "Integrated (Apple MPS)",
                    'api': 'MPS'
                })
            
            # Fallback for other GPUs (Vulkan/OpenCL)
            else:
                try:
                    vulkan_info = subprocess.check_output(['vulkaninfo'], text=True)
                    if 'GPU' in vulkan_info:
                        gpus.append({
                            'name': "Vulkan-compatible GPU",
                            'vram': "N/A (Vulkan detection limited)",
                            'type': "Generic (Vulkan/OpenCL)",
                            'api': 'Vulkan/OpenCL'
                        })
                except:
                    pass

        except Exception as e:
            print(f"GPU detection error: {e}")
        
        return gpus if gpus else [{
            'name': "No dedicated GPU detected",
            'vram': "N/A",
            'type': "CPU-only",
            'api': 'None'
        }]

    @staticmethod
    def get_os_info():
        os_map = {
            "Linux": "Linux",
            "Darwin": "Darwin",
            "Windows": "WindowsNT",
            "FreeBSD": "FreeBSD"
        }
        return os_map.get(platform.system(), "Unknown")

    @staticmethod
    def get_disk_info():
        disk = psutil.disk_usage('/')
        read_speed = 0.0
        start = time.time()
        with open(os.devnull, 'wb') as devnull:
            subprocess.run(['dd', 'if=/dev/zero', 'of=testfile', 'bs=1M', 'count=1024'], 
                          stdout=devnull, stderr=devnull)
            read_time = time.time() - start
            read_speed = (1024 * 1024 * 1024) / read_time  # 1GB in bytes
        os.remove('testfile')
        return {
            'total': f"{disk.total / 1e9:.2f} GB",
            'available': f"{disk.free / 1e9:.2f} GB",
            'read_speed': f"{read_speed / 1e6:.2f} MB/s"
        }

    @staticmethod
    def get_accelerator_info():
        accelerators = []
        
        # NVIDIA CUDA
        if torch.cuda.is_available():
            accelerators.append("CUDA GPU")
        
        # AMD ROCm
        if 'ROCM_PATH' in os.environ:
            accelerators.append("AMD ROCm")
        
        # Intel oneAPI
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            accelerators.append("OneAPI Intel XPU")
        
        # Apple MPS
        if torch.backends.mps.is_available():
            accelerators.append("MPSAccelerator (Apple)")
        
        # Vulkan (experimental)
        if shutil.which('vulkaninfo'):
            accelerators.append("Vulkan")
        
        # OpenCL (community support)
        if os.path.exists('/etc/OpenCL/vendors'):
            accelerators.append("OpenCL")
        
        return accelerators if accelerators else ["None detected"]

    @staticmethod
    def get_vram_type():
        if platform.system() == "Darwin":
            return "Unified Memory Architecture"
        elif torch.cuda.is_available():
            return "Dedicated (NVIDIA)"
        return "Integrated"

    @classmethod
    def generate_startup_banner(cls):
        info = {
            "CPU": cls.get_cpu_info(),
            "Memory": cls.get_memory_info(),
            "GPU": cls.get_gpu_info(),
            "OS Kernel": cls.get_os_info(),
            "Disk": cls.get_disk_info(),
            "Accelerators": cls.get_accelerator_info(),
            "VRAM Type": cls.get_vram_type()
        }
        
        # Generate warnings based on system specs
        warnings = []
        
        # CPU Architecture Warning
        if info['CPU']['architecture'] not in ['x86_64', 'AMD64']:
            warnings.append(
                f"⚠️ CPU Architecture Warning: {info['CPU']['architecture']} detected. "
                "This system may experience memory stability issues under ML workloads. "
                "Optimal performance requires AMD64 architecture with strong memory ordering [[1]][[2]]."
            )
        
        # AMD CPU Warning
        elif 'AMD' in info['CPU']['name']:
            warnings.append(
                f"⚠️ AMD CPU Warning: {info['CPU']['name']} detected. "
                "AMD processors may exhibit instability under sustained high ML workloads [[3]][[4]]."
            )
        
        # Intel Generation Warning
        else:
            intel_gen_match = re.search(r'\b(12th|13th|14th) Gen\b', info['CPU']['name'])
            if intel_gen_match:
                warnings.append(
                    f"⚠️ Intel CPU Warning: {intel_gen_match.group()} processor detected. "
                    "Recent Intel generations may have compatibility issues with CUDA optimizations [[5]]."
                )
        
        # GPU Warning
        if not any(gpu.get('api', '') == 'CUDA' for gpu in info['GPU']):
            warnings.append(
                "⚠️ GPU Compatibility Warning: Non-NVIDIA GPU detected. "
                "The system will miss CUDA-specific optimizations critical for ML performance [[6]][[7]]."
            )
        
        # Dedicated VRAM Check
        has_dedicated_vram = any(
            gpu.get('type', '') == "Dedicated (NVIDIA CUDA)" and 
            float(gpu.get('vram', '0').split()[0]) >= 24
            for gpu in info['GPU']
        )
        if not has_dedicated_vram:
            warnings.append(
                "⚠️ VRAM Capacity Warning: Insufficient dedicated VRAM (minimum 24GB required). "
                f"Current VRAM: {info['GPU'][0]['vram'] if info['GPU'] else 'N/A'} [[8]]."
            )
        
        # Disk Space Warning
        if float(info['Disk']['available'].split()[0]) < 40:
            warnings.append(
                "⚠️ Storage Warning: Low disk space detected. "
                f"Available: {info['Disk']['available']} (Minimum 80GB recommended) [[9]]."
            )
        
        # Disk Speed Warning
        if float(info['Disk']['read_speed'].split()[0]) < 9:
            warnings.append(
                "⚠️ Disk Performance Warning: Slow storage detected. "
                f"Read speed: {info['Disk']['read_speed']} (Minimum 9GB/s required). "
                "This may cause severe latency issues during model streaming operations [[10]]."
            )
        
        # Memory Capacity Warning
        if float(info['Memory']['available'].split()[0]) < 7:
            warnings.append(
                "⚠️ Memory Warning: Insufficient RAM available. "
                f"Available: {info['Memory']['available']} (Minimum 8GB recommended) [[11]]."
            )
        
        # Unified Memory Warning
        if info['VRAM Type'] == "Unified Memory Architecture":
            warnings.append(
                "⚠️ Memory Architecture Warning: Unified memory detected. "
                "The system may experience allocation conflicts under heavy workloads [[12]]."
            )
        
        # Final warning if multiple issues
        # Store critical warning separately
        info['CriticalWarning'] = None
        if len(warnings) > 3:
            critical_warning = (
                f"{Fore.RED}{Style.BRIGHT}⚠️ CRITICAL WARNING: Multiple hardware limitations detected. "
                f"This configuration may cause instability or performance degradation. Proceed with caution "
                f"to avoid excessive hardware stress [[13]].{Style.RESET_ALL}"
            )
            info['CriticalWarning'] = critical_warning
        info['Warnings'] = warnings
        return info

class EmbeddingThread(Thread):
    def __init__(self, embedding_model_path, ctx_window_llm, n_batch, database_manager):
        super().__init__(daemon=True)
        self.embedding_model_path = embedding_model_path
        self.ctx_window_llm = ctx_window_llm
        self.n_batch = n_batch
        self.embedding_queue = []  # Queue for text chunks to embed
        self.queue_lock = threading.Lock()
        self.database_manager = database_manager  # Store DatabaseManager
        self.embedding_llm = None  # Initialize within the thread
        self._stop_event = threading.Event()  # Stop event.

    def run(self):
        """Thread's main loop."""
        self.embedding_llm = Llama(model_path=self.embedding_model_path, n_ctx=self.ctx_window_llm, n_gpu_layers=-1,
                                   n_batch=self.n_batch, embedding=True)
        while not self._stop_event.is_set():
            with self.queue_lock:
                if not self.embedding_queue:
                    time.sleep(0.1)  # Avoid busy-waiting
                    continue
                task = self.embedding_queue.pop(0)
            text_chunk, slot, requester_type, doc_id = task

            try:
                embedding = self.embedding_llm.embed(text_chunk)
                # --- Robustness: Handle float or list ---
                if isinstance(embedding, float):
                    embedding = [embedding]  # Wrap in a list
                elif not isinstance(embedding, list):
                    print(OutputFormatter.color_prefix(
                        f"Warning: Unexpected embedding type: {type(embedding)}. Skipping.", "Internal"))
                    continue

                if not all(isinstance(x, (int, float)) for x in embedding):
                    print(OutputFormatter.color_prefix("Warning: Embedding contains non-numeric values. Skipping.",
                                                       "Internal"))
                    continue

                if requester_type == "CoT":
                    table_name = "CoT_generateResponse_History"
                else:
                    table_name = "interaction_history"

                self.database_manager.db_writer.schedule_write(
                    f"INSERT INTO {table_name} (slot, doc_id, chunk, embedding) VALUES (?, ?, ?, ?)",
                    (slot, doc_id, text_chunk, pickle.dumps(embedding))
                )
                print(OutputFormatter.color_prefix(
                    f"Scheduled database write for embedding: slot {slot}, type {requester_type}", "Internal"))
            except Exception as e:
                print(OutputFormatter.color_prefix(f"Error in embedding thread: {e}", "Internal"))
                traceback.print_exc()

    def stop(self):
        """Stops the embedding thread."""
        self._stop_event.set()

    def embed_and_store(self, text_chunk, slot, requester_type, doc_id):
        """Adds a text chunk to the embedding queue."""
        with self.queue_lock:
            self.embedding_queue.append((text_chunk, slot, requester_type, doc_id))


# --- GGUF Parsing (Adapted from reference code) ---
class GGUFValueType:  # Define outside the class
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


GGUF_MAGIC = b"GGUF"


class GGUFParser:
    def __init__(self, gguf_file_path: str):
        self.gguf_file_path = gguf_file_path
        self.metadata: Dict[str, Any] = {}
        self.tensor_infos: List[Tuple[str, List[int], int, int]] = []
        self.tokens: List[str] = []  # Store tokens here
        self.token_scores: List[float] = []  # and scores
        self.token_types: List[int] = []
        self._parse()

    def _read_bytes(self, file: BinaryIO, num_bytes: int) -> bytes:
        data = file.read(num_bytes)
        if len(data) != num_bytes:
            raise ValueError(f"Expected {num_bytes} bytes, but got {len(data)}.")
        return data

    def _read_u8(self, file: BinaryIO) -> int:
        return struct.unpack("<B", self._read_bytes(file, 1))[0]

    def _read_i8(self, file: BinaryIO) -> int:
        return struct.unpack("<b", self._read_bytes(file, 1))[0]

    def _read_u16(self, file: BinaryIO) -> int:
        return struct.unpack("<H", self._read_bytes(file, 2))[0]

    def _read_i16(self, file: BinaryIO) -> int:
        return struct.unpack("<h", self._read_bytes(file, 2))[0]

    def _read_u32(self, file: BinaryIO) -> int:
        return struct.unpack("<I", self._read_bytes(file, 4))[0]

    def _read_i32(self, file: BinaryIO) -> int:
        return struct.unpack("<i", self._read_bytes(file, 4))[0]

    def _read_u64(self, file: BinaryIO) -> int:
        return struct.unpack("<Q", self._read_bytes(file, 8))[0]

    def _read_i64(self, file: BinaryIO) -> int:
        return struct.unpack("<q", self._read_bytes(file, 8))[0]

    def _read_f32(self, file: BinaryIO) -> float:
        return struct.unpack("<f", self._read_bytes(file, 4))[0]

    def _read_f64(self, file: BinaryIO) -> float:
        return struct.unpack("<d", self._read_bytes(file, 8))[0]

    def _read_bool(self, file: BinaryIO) -> bool:
        return self._read_u8(file) != 0

    def _read_string(self, file: BinaryIO) -> str:
        length = self._read_u64(file)
        data = self._read_bytes(file, length)
        return data.decode("utf-8", errors="replace")

    def _read_array(self, file: BinaryIO) -> List[Any]:
        value_type = self._read_u32(file)
        length = self._read_u64(file)
        return [self._read_value(file, value_type) for _ in range(length)]

    def _read_value(self, file: BinaryIO, value_type: int) -> Any:
        if value_type == GGUFValueType.UINT8:
            return self._read_u8(file)
        elif value_type == GGUFValueType.INT8:
            return self._read_i8(file)
        elif value_type == GGUFValueType.UINT16:
            return self._read_u16(file)
        elif value_type == GGUFValueType.INT16:
            return self._read_i16(file)
        elif value_type == GGUFValueType.UINT32:
            return self._read_u32(file)
        elif value_type == GGUFValueType.INT32:
            return self._read_i32(file)
        elif value_type == GGUFValueType.FLOAT32:
            return self._read_f32(file)
        elif value_type == GGUFValueType.UINT64:
            return self._read_u64(file)
        elif value_type == GGUFValueType.INT64:
            return self._read_i64(file)
        elif value_type == GGUFValueType.FLOAT64:
            return self._read_f64(file)
        elif value_type == GGUFValueType.BOOL:
            return self._read_bool(file)
        elif value_type == GGUFValueType.STRING:
            return self._read_string(file)
        elif value_type == GGUFValueType.ARRAY:
            return self._read_array(file)
        else:
            raise ValueError(f"Invalid GGUF value type: {value_type}")

    def _read_metadata(self, file: BinaryIO) -> Dict[str, Any]:
        metadata = {}
        metadata_count = self._read_u64(file)
        for _ in range(metadata_count):
            key = self._read_string(file)
            value_type = self._read_u32(file)
            value = self._read_value(file, value_type)
            metadata[key] = value
        return metadata

    def _read_tensor_info(self, file: BinaryIO) -> Tuple[str, List[int], int, int]:
        name = self._read_string(file)
        n_dims = self._read_u32(file)
        dims = [self._read_u64(file) for _ in range(n_dims)]
        tensor_type = self._read_u32(file)
        offset = self._read_u64(file)
        return name, dims, tensor_type, offset

    def _parse(self):
        """Parses the GGUF file and populates metadata and tensor_infos."""
        try:
            with open(self.gguf_file_path, "rb") as f:
                magic = self._read_bytes(f, 4)
                if magic != GGUF_MAGIC:
                    raise ValueError("Invalid GGUF file (magic number mismatch).")

                version = self._read_u32(f)
                tensor_count = self._read_u64(f)

                self.metadata = self._read_metadata(f)

                for _ in range(tensor_count):
                    self.tensor_infos.append(self._read_tensor_info(f))
        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error parsing GGUF file: {e}", "Internal"))  # Use consistent logging
            self.metadata = {}  # Reset to empty dict on error
            self.tensor_infos = []

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the parsed metadata."""
        return self.metadata

    def get_tensor_infos(self) -> List[Tuple[str, List[int], int, int]]:
        """Returns the parsed tensor information."""
        return self.tensor_infos


class DatabaseManager:
    def __init__(self, db_file, loop):
        self.db_connection = sqlite3.connect(db_file, check_same_thread=False)
        self.db_cursor = self.db_connection.cursor()
        self.loop = loop
        self.db_writer = DatabaseWriter(self.db_connection, self.loop)
        self.db_writer.start_writer()
        self._initialize_database()

    def _initialize_database(self):
        """Creates necessary tables if they don't exist."""
        self.db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS interaction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slot INTEGER,
                role TEXT,
                message TEXT,
                response TEXT,
                context_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                doc_id TEXT,  -- Add doc_id
                chunk TEXT,   -- Add chunk
                embedding BLOB
            )
            """
        )
        self.db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS CoT_generateResponse_History (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slot INTEGER,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_learning_context_embedding (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slot INTEGER,
                doc_id TEXT,
                chunk TEXT,
                embedding BLOB
            )
            """
        )
        # Create task_queue table
        self.db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS task_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT,
                args TEXT,  -- Store arguments as JSON string
                priority INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.db_connection.commit()

    @property
    def ai_runtime_manager(self):
        return self._ai_runtime_manager  # Ensure this attribute is set

    @ai_runtime_manager.setter
    def ai_runtime_manager(self, value):
        self._ai_runtime_manager = value

    def save_task_queue(self, task_queue, backbrain_tasks, mesh_network_tasks):
        """Saves the current state of the task queues to the database."""
        try:
            # Clear the existing queue
            self.db_cursor.execute("DELETE FROM task_queue")

            # Save the main task queue.  Iterate directly over the combined (task, args), priority tuples.
            for (task, args), priority in task_queue:
                task_name = task.__name__ if callable(task) else str(task)  # Handle non-callable tasks
                args_str = json.dumps(args)
                self.db_writer.schedule_write(
                    "INSERT INTO task_queue (task_name, args, priority) VALUES (?, ?, ?)",
                    (task_name, args_str, priority)
                )

            # Save the backbrain task queue.  Same iteration pattern.
            for (task, args), priority in backbrain_tasks:
                task_name = task.__name__ if callable(task) else str(task)  # Handle non-callable tasks
                args_str = json.dumps(args)
                self.db_writer.schedule_write(
                    "INSERT INTO task_queue (task_name, args, priority) VALUES (?, ?, ?)",
                    (task_name, args_str, priority)
                )

            # Save meshNetworkProcessingIO Queue. Same iteration pattern.
            if hasattr(self, 'mesh_network_tasks'):
                for (task, args), priority in self.mesh_network_tasks:
                    task_name = task.__name__ if callable(task) else str(task)  # Handle non-callable tasks
                    args_str = json.dumps(args)
                    self.db_writer.schedule_write(
                        "INSERT INTO task_queue (task_name, args, priority) VALUES (?, ?, ?)",
                        (task_name, args_str, priority)
                    )
            print(OutputFormatter.color_prefix("Task queue saved to database.", "Internal"))

        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error saving task queue: {e}", "Internal"))

    def load_task_queue(self):  # modified to load the new task queue.
        """Loads the task queue from the database."""
        try:
            self.db_cursor.execute("SELECT task_name, args, priority FROM task_queue ORDER BY created_at ASC")
            rows = self.db_cursor.fetchall()

            task_queue = []
            backbrain_tasks = []
            mesh_network_tasks = []  # Initialize mesh_network_tasks
            for task_name, args_str, priority in rows:
                args = json.loads(args_str)

                # Resolve the task function from its name
                if task_name == "generate_response":
                    task_callable = self.ai_runtime_manager.generate_response
                elif task_name == "process_branch_prediction_slot":
                    task_callable = self.ai_runtime_manager.process_branch_prediction_slot
                # Add more task name to function mappings as needed
                else:
                    print(OutputFormatter.color_prefix(f"Unknown task name found in database: {task_name}", "Internal"))
                    continue

                task = (task_callable, args)
                if priority == 3:
                    backbrain_tasks.append((task, priority))
                elif priority == 99:  # Load meshNetworkProcessingIO tasks
                    mesh_network_tasks.append((task, priority))
                else:
                    task_queue.append((task, priority))

            print(OutputFormatter.color_prefix("Task queue loaded from database.", "Internal"))
            # Initialize mesh_network_tasks if it doesn't exist
            if not hasattr(self, 'mesh_network_tasks'):
                self.mesh_network_tasks = []

            self.mesh_network_tasks = mesh_network_tasks  # Assign the loaded tasks

            return task_queue, backbrain_tasks

        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error loading task queue: {e}", "Internal"))
            return [], []

    def print_table_contents(self, table_name):
        """Prints the contents of a specified table."""
        print(OutputFormatter.color_prefix(f"--- Contents of table: {table_name} ---", "Internal"))
        try:
            self.db_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [column[1] for column in self.db_cursor.fetchall()]
            print(OutputFormatter.color_prefix(f"Columns: {', '.join(columns)}", "Internal"))

            self.db_cursor.execute(f"SELECT * FROM {table_name}")
            rows = self.db_cursor.fetchall()
            if rows:
                for row in rows:
                    print(OutputFormatter.color_prefix(row, "Internal"))
            else:
                print(OutputFormatter.color_prefix("Table is empty.", "Internal"))
        except sqlite3.OperationalError as e:
            print(OutputFormatter.color_prefix(f"Error reading table {table_name}: {e}", "Internal"))
        print(OutputFormatter.color_prefix("--- End of table ---", "Internal"))

    def get_chat_history(self, slot):
        """Retrieves chat history for a specific slot."""
        try:
            self.db_cursor.execute(
                "SELECT role, message FROM interaction_history WHERE slot=? ORDER BY timestamp ASC",
                (slot,),
            )
            history = self.db_cursor.fetchall()
            return history
        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error retrieving chat history: {e}", "Internal"))
            return []

    def fetch_chat_history(self, slot):
        """Fetches chat history for a specific slot and formats it for the prompt."""
        self.db_cursor.execute(
            "SELECT role, message FROM interaction_history WHERE slot=? ORDER BY timestamp",
            (slot,),
        )
        rows = self.db_cursor.fetchall()
        # Corrected line, filter out None values
        history = [{"role": role.lower(), "content": message} for role, message in rows if role is not None]
        return history

    async def async_db_write(self, slot, query, response):
        """Asynchronously writes user queries and AI responses to the database."""
        try:
            doc_id = str(time.time())  # Generate a unique doc_id
            self.db_writer.schedule_write(  # Modified to include chunk and doc_id.
                "INSERT INTO interaction_history (slot, role, message, response, context_type, doc_id, chunk, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (slot, "User", query, response, "main", doc_id, query, pickle.dumps([])),
                # Add doc_id and chunk with empty embedding initially
            )
            self.db_writer.schedule_write(  # Modified to include chunk and doc_id.
                "INSERT INTO interaction_history (slot, role, message, response, context_type, doc_id, chunk, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (slot, "AI", response, "", "main", doc_id, response, pickle.dumps([])),
                # Add doc_id and chunk with empty embedding initially
            )
        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error scheduling write to database: {e}", "Internal"))

    def close(self):
        """Closes the database connection and stops the writer task."""
        self.db_writer.close()


class DatabaseWriter:
    def __init__(self, db_connection, loop):
        self.db_connection = db_connection
        self.db_cursor = db_connection.cursor()
        self.write_queue = asyncio.Queue()
        self.loop = loop
        self.writer_task = None

    def start_writer(self):
        """Starts the writer task."""
        self.writer_task = self.loop.create_task(self._writer())

    async def _writer(self):
        while True:
            print(OutputFormatter.color_prefix(f"DatabaseWriter: Waiting for write operation...",
                                               "Internal"))  # Added Logging
            try:
                write_operation = await asyncio.wait_for(self.write_queue.get(), timeout=5.0)  # 5-second timeout
                if write_operation is None:
                    print(OutputFormatter.color_prefix(f"DatabaseWriter: Received shutdown signal.",
                                                       "Internal"))  # Added Logging
                    break

                sql, data = write_operation
                print(OutputFormatter.color_prefix(f"DatabaseWriter: Executing SQL: {sql[:50]}...",
                                                   "Internal"))  # Added Logging
                self.db_cursor.execute(sql, data)
                print(
                    OutputFormatter.color_prefix(f"DatabaseWriter: Committing changes...", "Internal"))  # Added Logging
                self.db_connection.commit()
                print(OutputFormatter.color_prefix(f"Database write successful: {sql[:50]}...", "Internal"))

            except Exception as e:
                print(OutputFormatter.color_prefix(f"Error during database write: {str(e)}", "Internal"))
                self.db_connection.rollback()
            except asyncio.TimeoutError:  # Corrected Indentation: Same level as inner try
                print(OutputFormatter.color_prefix(f"DatabaseWriter: Timeout waiting for write operation.", "Internal"))
            finally:
                self.write_queue.task_done()

    def schedule_write(self, sql, data):
        """Schedules a write operation to be executed sequentially."""
        self.write_queue.put_nowait((sql, data))

    def close(self):
        """Stops the writer task and closes the database connection."""
        self.write_queue.put_nowait(None)  # Signal to stop
        if self.writer_task:
            self.writer_task.cancel()
        self.db_connection.close()


class OutputFormatter:

    @staticmethod
    def separator():
        return ["-"*20, "-"*20]

    @staticmethod
    def format_system_info(info):
        # Format main system table
        header = f"{Fore.CYAN}Adelaide and Albert Engine Startup System Initialization{Style.RESET_ALL}"
        table = []

        # CPU Section
        cpu_section = [
            ["CPU Architecture", info['CPU']['architecture']],
            ["CPU Model", info['CPU']['name']],
            ["Cores/Threads", f"{info['CPU']['cores']}/{info['CPU']['threads']}"]
        ]

        # Memory Section
        memory_section = [
            ["Total Memory", info['Memory']['total']],
            ["Available Memory", info['Memory']['available']],
            ["ECC Support", info['Memory']['ecc']]
        ]

        # GPU/VRAM Section
        gpu_section = []
        for gpu in info['GPU']:
            gpu_section.extend([
                ["GPU Model", gpu['name']],
                ["VRAM", gpu['vram']],
                ["VRAM Type", gpu.get('type', 'N/A')]  # Use .get() for safety
            ])

        # Storage Section
        storage_section = [
            ["Total Storage", info['Disk']['total']],
            ["Available Storage", info['Disk']['available']],
            ["Disk Read Speed", info['Disk']['read_speed']]
        ]

        # Accelerator Section
        accelerator_section = [
            ["Accelerators", ', '.join(info['Accelerators'])]
        ]

        # OS Section
        os_section = [
            ["Kernel Type", info['OS Kernel']]
        ]

        # Combine all sections with separators
        full_table = (
            cpu_section + [OutputFormatter.separator()] +
            memory_section + [OutputFormatter.separator()] +
            gpu_section + [OutputFormatter.separator()] +
            storage_section + [OutputFormatter.separator()] +
            accelerator_section + [OutputFormatter.separator()] +
            os_section
        )

        # Format warnings section
        warnings = info.get('Warnings', [])
        critical_warning = info.get('CriticalWarning', None)
        
        # Format regular warnings in yellow
        warning_messages = "\n".join([
            f"{Fore.YELLOW}{Style.BRIGHT}⚠️ {warning}{Style.RESET_ALL}"
            for warning in warnings
        ])
        
        # Add critical warning in red if present
        if critical_warning:
            warning_messages += f"\n{critical_warning}"
        
        return f"""
{header}
{tabulate(full_table, headers=['Component', 'Specification'], tablefmt='psql')}

{Fore.YELLOW}{Style.BRIGHT}System Warnings:{Style.RESET_ALL}
{warning_messages}
"""

    @staticmethod
    def get_username():
        """Gets the username of the current user."""
        try:
            import getpass
            return getpass.getuser()
        except ImportError:
            try:
                return os.getlogin()
            except OSError:
                return platform.node().split('.')[0]

    @staticmethod
    def color_prefix(text, prefix_type, generation_time=None, progress=None, token_count=None, slot=None):
        """Formats text with colors and additional information."""
        reset = attr('reset')
        if prefix_type == "User":
            username = OutputFormatter.get_username()
            return f"{fg(202)}{username}{reset} {fg(172)}⚡{reset} {fg(196)}×{reset} {fg(166)}⟩{reset} {text}"
        elif prefix_type == "Adelaide":
            # Retrieve slot from the context within generate_response
            # This assumes you can somehow determine the slot from within generate_response
            # You might need to pass slot as an argument to color_prefix or retrieve it from a global/shared context

            # For demonstration, let's assume you have a way to get the slot like this:

            context_length = ai_runtime_manager.calculate_total_context_length(slot, "main")  # Use
            if generation_time is not None and token_count is not None:
                return (
                    f"{fg(99)}Adelaide{reset} {fg(105)}⚡{reset} {fg(111)}×{reset} "
                    f"{fg(117)}⟨{context_length}⟩{reset} {fg(123)}⟩{reset} "
                    f"{fg(250)}({generation_time:.2f}s){reset} {fg(250)}({token_count:.2f} tokens){reset} {text}"
                )
            elif generation_time is not None:
                return (
                    f"{fg(99)}Adelaide{reset} {fg(105)}⚡{reset} {fg(111)}×{reset} "
                    f"{fg(117)}⟨{context_length}⟩{reset} {fg(123)}⟩{reset} "
                    f"{fg(250)}({generation_time:.2f}s){reset} {text}"
                )
            else:
                return (
                    f"{fg(99)}Adelaide{reset} {fg(105)}⚡{reset} {fg(111)}×{reset} "
                    f"{fg(117)}⟨{context_length}⟩{reset} {fg(123)}⟩{reset} "
                    f"{text}"
                )
        elif prefix_type == "Internal":
            if progress is not None and generation_time is not None and token_count is not None:
                return (
                    f"{fg(135)}Ιnternal{reset} {fg(141)}⊙{reset} {fg(147)}○{reset} "
                    f"{fg(183)}⟩{reset} {fg(177)}[{progress:.1f}%]{reset} {fg(250)}({generation_time:.2f}s, {token_count} tokens){reset} {text}"
                )
            elif progress is not None and generation_time is not None:
                return (
                    f"{fg(135)}Ιnternal{reset} {fg(141)}⊙{reset} {fg(147)}○{reset} "
                    f"{fg(183)}⟩{reset} {fg(177)}[{progress:.1f}%]{reset} {fg(250)}({generation_time:.2f}s){reset} {text}"
                )
            elif progress is not None:
                return (
                    f"{fg(135)}Ιnternal{reset} {fg(141)}⊙{reset} {fg(147)}○{reset} "
                    f"{fg(183)}⟩{reset} {fg(177)}[{progress:.1f}%]{reset} {text}"
                )
            else:
                return (
                    f"{fg(135)}Ιnternal{reset} {fg(141)}⊙{reset} {fg(147)}○{reset} "
                    f"{fg(183)}⟩{reset} {fg(177)} {text}{reset}"
                )
        elif prefix_type == "BackbrainController":
            if generation_time is not None:
                return (
                    f"{fg(153)}Βackbrain{reset} {fg(195)}∼{reset} {fg(159)}≡{reset} "
                    f"{fg(195)}⟩{reset} {fg(250)}({generation_time:.2f}s){reset} {text}"
                )
            else:
                return (
                    f"{fg(153)}Βackbrain{reset} {fg(195)}∼{reset} {fg(159)}≡{reset} "
                    f"{fg(195)}⟩{reset} {text}"
                )
        elif prefix_type == "branch_predictor":  # Special prefix for branch_predictor
            return (
                f"{fg(220)}branch_predictor{reset} {fg(221)}∼{reset} {fg(222)}≡{reset} "
                f"{fg(223)}⟩{reset} {text}"
            )
        elif prefix_type == "Watchdog":
            return (
                f"{fg(243)}Watchdog{reset} {fg(244)}⚯{reset} {fg(245)}⊜{reset} "
                f"{fg(246)}⟩{reset} {text}"
            )
        elif prefix_type == "ServerOpenAISpec":
            return (
                f"{fg(40)}ServerOpenAISpec{reset} {fg(41)}➤{reset} {fg(42)}➤{reset} {fg(43)}➤{reset} {text}"
            )
        else:
            return text


class ChatFormatter:  # Renamed to be more generic
    def __init__(self, gguf_parser=None):
        self.gguf_parser = gguf_parser  # Store the parser (might be None initially)
        self.default_template_string = """
        {% if messages[0]['role'] == 'system' %}
            {% set offset = 1 %}
        {% else %}
            {% set offset = 0 %}
        {% endif %}

        {% for message in messages %}
            {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
                {% if loop.index0 == 0 %}
                    {{ '<|user|>\n' + message['content'] | trim + '<|end|>\n' }}
                {% else %}
                    {{ 'Error: Conversation roles must alternate user/assistant/user/assistant/...' }}
                {% endif %}
            {% else %}
                {{ '<|' + message['role'] + '|>\n' + message['content'] | trim + '<|end|>\n' }}
            {% endif %}
        {% endfor %}

        {% if add_generation_prompt %}
            {{ '<|assistant|>\n' }}
        {% endif %}
        """
        self.chatml_template = Template(self.default_template_string)
        # Don't initialize self.template here!
        self.template = None  # Initialize to None

    def _get_template(self) -> Template:
        """Gets the appropriate Jinja2 template, prioritizing GGUF metadata."""
        if self.gguf_parser:
            metadata = self.gguf_parser.get_metadata()
            chat_template_str = metadata.get("tokenizer.chat_template")

            if chat_template_str:
                if isinstance(chat_template_str, bytes):
                    chat_template_str = chat_template_str.decode('utf-8', errors='replace')
                try:
                    print(OutputFormatter.color_prefix(f"Using chat template from GGUF metadata:\n{chat_template_str}",
                                                       "Internal"))
                    return Template(chat_template_str)
                except Exception as e:
                    print(OutputFormatter.color_prefix(
                        f"Error creating template from GGUF metadata: {e}. Falling back to ChatML.", "Internal"))
                    # Fallback to ChatML is handled below

        print(OutputFormatter.color_prefix("Using default ChatML template.", "Internal"))
        return self.chatml_template  # Return the ChatML template

    def create_prompt(self, messages, add_generation_prompt=True):
        """
        Creates a prompt using the determined template.  Template is lazily initialized.
        """
        if self.template is None:  # Lazily initialize the template
            self.template = self._get_template()

        return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt)


class Watchdog:
    def __init__(self, restart_script_path, ai_runtime_manager):
        self.restart_script_path = restart_script_path
        self.ai_runtime_manager = ai_runtime_manager
        self.loop = None  # Do not initialize the loop here

    async def monitor(self):
        """Monitors the system and triggers model unloading on timeout."""
        while True:
            try:
                await asyncio.sleep(0.095)

                if self.ai_runtime_manager.last_task_info:
                    task_name = self.ai_runtime_manager.last_task_info["task"].__name__
                    elapsed_time = self.ai_runtime_manager.last_task_info["elapsed_time"]

                    if task_name == "generate_response" and elapsed_time > 60:
                        print(OutputFormatter.color_prefix(
                            "Watchdog detected potential issue: generate_response timeout",
                            "Watchdog",
                        ))
                        # --- Corrected: Only unload the model ---
                        self.ai_runtime_manager.unload_model()
                        # --- No longer exiting the monitor loop ---

            except Exception as e:
                print(OutputFormatter.color_prefix(f"Watchdog error: {e}", "Watchdog"))

    def restart(self):
        """Restarts the program."""
        print(OutputFormatter.color_prefix("Restarting program...", "Watchdog"))
        python = sys.executable
        os.execl(python, python, self.restart_script_path)

    def start(self, loop):
        self.loop = loop
        self.loop.create_task(self.monitor())


class AIModelPreRuntimeManager:
    """
    Manages pre-runtime preparation of AI models.
    """

    @staticmethod
    def check_tool_integrity(tool_dir: str, essential_files: list) -> bool:
        """
        Checks if the essential files for a tool exist.
        """
        if not os.path.exists(tool_dir):
            return False
        for file in essential_files:
            if not os.path.exists(os.path.join(tool_dir, file)):
                return False
        return True

    @staticmethod
    def cloning_tools():
        """
        Clones necessary repositories, builds llama-quantize, and manages updates.
        """
        library_dir = "./Library"
        os.makedirs(library_dir, exist_ok=True)

        repositories = {
            "llama.cpp": {
                "url": "https://github.com/ggerganov/llama.cpp.git",
                "essential_files": ["convert_hf_to_gguf.py", "CMakeLists.txt", "llama.h"],
            },
            "stable-diffusion.cpp": {
                "url": "https://github.com/leejet/stable-diffusion.cpp.git",
                "essential_files": ["stablediffusion.cpp", "CMakeLists.txt", "common.h"]
            },
        }

        for repo_name, repo_info in repositories.items():
            target_dir = os.path.join(library_dir, repo_name)
            repo_url = repo_info["url"]
            essential_files = repo_info["essential_files"]
            last_update_file = os.path.join(target_dir, ".last_update")

            if AIModelPreRuntimeManager.check_tool_integrity(target_dir, essential_files):
                print(f"{repo_name} found in {target_dir} and appears to be complete.")

                needs_update = True
                if os.path.exists(last_update_file):
                    try:
                        with open(last_update_file, "r") as f:
                            last_update_str = f.read().strip()
                            last_update = datetime.fromisoformat(last_update_str)
                            if datetime.now() - last_update < timedelta(days=7):
                                needs_update = False
                                print(f"{repo_name} was updated recently. Skipping update.")
                    except (ValueError, OSError):
                        print(f"Error reading last update time for {repo_name}.  Will attempt update.")

                if needs_update:
                    print(f"Updating {repo_name} in {target_dir}...")
                    try:
                        command = ["git", "pull", "--depth=1", "origin", "master"]
                        subprocess.run(command, check=True, cwd=target_dir)

                        with open(last_update_file, "w") as f:
                            f.write(datetime.now().isoformat())
                        print(f"{repo_name} updated successfully.")

                        if repo_name == "llama.cpp":
                            AIModelPreRuntimeManager.build_llama_quantize()
                            AIModelPreRuntimeManager.install_llama_cpp_requirements()  # NEW

                    except subprocess.CalledProcessError as e:
                        print(f"Error updating {repo_name}: {e}")
                    except FileNotFoundError:
                        print("Error: 'git' command not found.")
                continue

            if not os.path.exists(target_dir):
                print(f"Cloning {repo_name} from {repo_url} to {target_dir}...")
                command = ["git", "clone", "--depth=1", repo_url, target_dir]
                try:
                    subprocess.run(command, check=True, cwd=os.getcwd())
                    with open(last_update_file, "w") as f:
                        f.write(datetime.now().isoformat())
                    print(f"Successfully cloned {repo_name}.")

                    if repo_name == "llama.cpp":
                        AIModelPreRuntimeManager.build_llama_quantize()
                        AIModelPreRuntimeManager.install_llama_cpp_requirements()  # NEW

                except subprocess.CalledProcessError as e:
                    print(f"Error cloning {repo_name}: {e}")
                except FileNotFoundError:
                    print("Error: 'git' command not found.")

        quantize_tool_path = os.path.join("./Library", "llama.cpp", "build", "bin", "llama-quantize")
        if not os.path.exists(quantize_tool_path):
            quantize_tool_path = os.path.join("./Library", "llama.cpp", "build", "llama-quantize")
            if not os.path.exists(quantize_tool_path):
                AIModelPreRuntimeManager.build_llama_quantize()
            else:
                print("llama-quantize already built. Skipping build process.")
        else:
            print("llama-quantize already built. Skipping build process.")

    @staticmethod
    def install_llama_cpp_requirements():
        """Installs llama.cpp requirements, skipping version pinning."""
        llama_cpp_path = os.path.join("./Library", "llama.cpp")
        requirements_path = os.path.join(llama_cpp_path, "requirements.txt")

        if not os.path.exists(requirements_path):
            print("requirements.txt not found in llama.cpp directory. Skipping installation.")
            return

        try:
            print("Installing llama.cpp requirements (without version pinning)...")

            # Read the requirements file
            with open(requirements_path, 'r') as f:
                requirements = f.readlines()

            # Filter out lines with == or >=
            filtered_requirements = [
                line.strip() for line in requirements
                if not re.search(r'[=><]=', line)  # Use regex to check for =, >, <
            ]

            # Install the filtered requirements
            if filtered_requirements:  # Only install if there are requirements left
                subprocess.run(["pip", "install"] + filtered_requirements, check=True)
                print("llama.cpp requirements installed successfully (without version pinning).")
            else:
                print("No requirements to install after filtering.")

        except subprocess.CalledProcessError as e:
            print(f"Error installing llama.cpp requirements: {e}")
        except FileNotFoundError:
            print("Error: pip command not found.")

    @staticmethod
    def build_llama_quantize():
        """Builds the llama-quantize binary."""
        llama_cpp_path = os.path.join("./Library", "llama.cpp")
        build_dir = os.path.join(llama_cpp_path, "build")
        if not os.path.exists(llama_cpp_path):
            raise FileNotFoundError("llama.cpp directory not found.  Run cloning_tools() first.")

        try:
            print("Building llama-quantize...")
            os.makedirs(build_dir, exist_ok=True)
            cmake_config_command = ["cmake", "-B", "build"]
            subprocess.run(cmake_config_command, check=True, cwd=llama_cpp_path)
            cmake_build_command = ["cmake", "--build", "build", "--config", "Release"]
            subprocess.run(cmake_build_command, check=True, cwd=llama_cpp_path)
            print("llama-quantize built successfully.")

        except subprocess.CalledProcessError as e:
            print(f"Error building llama-quantize: {e}")
        except FileNotFoundError:
            print("Error: cmake command not found")

    @staticmethod
    def download_model(repo_id: str, local_dir: str, revision: str = None, **kwargs):
        """Downloads a model from the Hugging Face Hub."""
        os.makedirs(local_dir, exist_ok=True)
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo_id, revision=revision)

        for file in repo_files:
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    revision=revision,
                    **kwargs
                )
            except Exception as e:
                print(f"Error downloading {file}: {e}")

    @staticmethod
    def convert_to_gguf(source_dir: str, target_dir: str, model_name: str):
        """
        Converts a Hugging Face model to GGUF format and then quantizes it.
        """
        os.makedirs(target_dir, exist_ok=True)

        llama_cpp_path = os.path.join("./Library", "llama.cpp")
        convert_script_path = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
        target_gguf_path_f16 = os.path.join(target_dir, f"{model_name}.tmp")

        if not os.path.exists(convert_script_path):
            raise FileNotFoundError(
                f"convert_hf_to_gguf.py not found at {convert_script_path}. Run cloning_tools() first.")

        command = [
            "python3",
            convert_script_path,
            source_dir,
            "--outfile", target_gguf_path_f16,
            "--outtype", "f16"
        ]
        try:
            print(f"Converting {source_dir} to GGUF (f16) format...")
            subprocess.run(command, check=True)
            print(f"Successfully converted model to {target_gguf_path_f16}")
        except subprocess.CalledProcessError as e:
            print(f"Error during conversion to GGUF: {e}")
            return
        except FileNotFoundError:
            print(f"Error: convert_hf_to_gguf.py not found. Ensure cloning_tools() has run.")
            return

        quantize_tool_path = os.path.join("./Library", "llama.cpp", "build", "bin", "llama-quantize")
        if not os.path.exists(quantize_tool_path):
            quantize_tool_path = os.path.join("./Library", "llama.cpp", "build", "llama-quantize")
            if not os.path.exists(quantize_tool_path):
                raise FileNotFoundError("llama-quantize executable not found.  Build it using build_llama_quantize().")

        target_gguf_path_quantized = os.path.join(target_dir, f"{model_name}.gguf")
        quantize_command = [
            quantize_tool_path,
            target_gguf_path_f16,
            target_gguf_path_quantized,
            "Q4_K_M"
        ]

        try:
            print(f"Quantizing {target_gguf_path_f16} to {target_gguf_path_quantized}...")
            subprocess.run(quantize_command, check=True)
            print(f"Successfully quantized model to {target_gguf_path_quantized}")
            os.remove(target_gguf_path_f16)
            print(f"Temporary file {target_gguf_path_f16} removed.")
        except subprocess.CalledProcessError as e:
            print(f"Error during quantization: {e}")
        except FileNotFoundError as e:
            print(f"Error: {e}. Ensure you have built llama-quantize.")

    @staticmethod
    def prepare_llm_model(repo_id: str, model_name: str, revision: str = None):
        """Downloads, converts, and quantizes an LLM."""
        source_dir = os.path.join("./Model", model_name)
        target_dir = os.path.join("./Model/ModelCompiledRuntime")
        AIModelPreRuntimeManager.download_model(repo_id, source_dir, revision=revision)
        AIModelPreRuntimeManager.convert_to_gguf(source_dir, target_dir, model_name)

        # LLaVA-specific post-processing (after llama.cpp is built)
        if "llava" in model_name.lower():  # Check if it's a LLaVA model
            print(f"Performing LLaVA-specific GGUF conversion for {model_name}...")
            llama_cpp_path = os.path.join("./Library", "llama.cpp")
            llava_examples_path = os.path.join(llama_cpp_path, "examples", "llava")
            requirements_path = os.path.join(llava_examples_path, "requirements.txt")

            # 1. Install requirements
            try:
                print("Installing LLaVA requirements...")
                subprocess.run(["pip", "install", "-r", requirements_path], check=True)
                print("LLaVA requirements installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing LLaVA requirements: {e}")
                return  # Or raise, depending on how critical this is

            # 2. Prepare CLIP model
            vit_dir = os.path.join(source_dir, "vit")
            os.makedirs(vit_dir, exist_ok=True)
            try:
                print("Preparing CLIP model for GGUF conversion...")
                shutil.copy(os.path.join(source_dir, "llava.clip"), os.path.join(vit_dir, "pytorch_model.bin"))
                shutil.copy(os.path.join(source_dir, "llava.projector"),
                            os.path.join(vit_dir, "llava.projector"))  # Correct file name
                config_vit_url = "https://huggingface.co/cmp-nct/llava-1.6-gguf/raw/main/config_vit.json"
                subprocess.run(["curl", "-s", "-q", config_vit_url, "-o", os.path.join(vit_dir, "config.json")],
                               check=True)
                print("CLIP model prepared successfully.")
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                print(f"Error preparing CLIP model: {e}")
                return

            # 3. Convert image encoder
            convert_script = os.path.join(llava_examples_path, "convert_image_encoder_to_gguf.py")
            convert_command = [
                "python3",
                convert_script,
                "-m", vit_dir,
                "--llava-projector", os.path.join(vit_dir, "llava.projector"),
                "--output-dir", vit_dir,
                "--clip-model-is-vision"
            ]
            try:
                print("Converting image encoder to GGUF...")
                subprocess.run(convert_command, check=True)
                print("Image encoder converted to GGUF successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error converting image encoder: {e}")
                return

            return os.path.join(target_dir, f"{model_name}.gguf")

    @staticmethod
    def prepare_embedding_model(repo_id: str, model_name: str, revision: str = None):
        """Downloads the embedding model (no conversion needed)."""
        source_dir = os.path.join("./Model", model_name)
        AIModelPreRuntimeManager.download_model(repo_id, source_dir, revision)
        gguf_files = glob.glob(os.path.join(source_dir, "*.gguf"))
        if gguf_files:
            return gguf_files[0]
        else:
            print(f"Warning: No .gguf file found in {source_dir}.")
            return None

    @staticmethod
    def prepare_stable_diffusion_model(repo_id: str, model_name: str, revision: str = None):
        """Downloads ONLY .safetensors files for Stable Diffusion model."""
        source_dir = os.path.join("./Model", model_name)
        
        # Modified download with .safetensors filtering
        AIModelPreRuntimeManager.download_model(
            repo_id,
            source_dir,
            revision=revision,
            allowed_extensions=['.safetensors']  # New parameter
        )
        
        # Verify .safetensors files exist
        safetensors_files = glob.glob(os.path.join(source_dir, "*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors files found in {source_dir}")
        
        print(f"Stable Diffusion model downloaded: {safetensors_files[0]}")
        return safetensors_files[0]  # Return path to .safetensors file

    @staticmethod
    def build_llama_quantize():
        """Builds the llama-quantize binary from the cloned llama.cpp repository."""
        llama_cpp_path = os.path.join("./Library", "llama.cpp")
        build_dir = os.path.join(llama_cpp_path, "build")
        # --- Check moved INSIDE build_llama_quantize ---
        if not os.path.exists(llama_cpp_path):
            raise FileNotFoundError("llama.cpp directory not found.  Run cloning_tools() first.")

        try:
            print("Building llama-quantize...")
            os.makedirs(build_dir, exist_ok=True)
            # --- cwd changed to llama_cpp_path ---
            cmake_config_command = ["cmake", "-B", "build"]
            subprocess.run(cmake_config_command, check=True, cwd=llama_cpp_path)  # Use llama_cpp_path
            # --- cwd changed to llama_cpp_path ---
            cmake_build_command = ["cmake", "--build", "build", "--config", "Release"]
            subprocess.run(cmake_build_command, check=True, cwd=llama_cpp_path)  # Use llama_cpp_path
            print("llama-quantize built successfully.")

        except subprocess.CalledProcessError as e:
            print(f"Error building llama-quantize: {e}")
        except FileNotFoundError:
            print("Error: cmake command not found")

    @staticmethod
    def download_model(repo_id: str, local_dir: str, revision: str = None, 
                      allowed_extensions: list = None, **kwargs):
        """Downloads model files, optionally filtering by extensions."""
        os.makedirs(local_dir, exist_ok=True)
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo_id, revision=revision)
        
        for file in repo_files:
            # Skip files that don't match allowed extensions
            if allowed_extensions and not any(file.endswith(ext) for ext in allowed_extensions):
                continue
                
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    revision=revision,
                    **kwargs
                )
            except Exception as e:
                print(f"Error downloading {file}: {e}")

    @staticmethod
    def convert_to_gguf(source_dir: str, target_dir: str, model_name: str):
        """
        Converts a Hugging Face model to GGUF format and then quantizes it.
        """
        os.makedirs(target_dir, exist_ok=True)

        llama_cpp_path = os.path.join("./Library", "llama.cpp")
        convert_script_path = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
        target_gguf_path_f16 = os.path.join(target_dir, f"{model_name}.tmp")

        if not os.path.exists(convert_script_path):
            raise FileNotFoundError(
                f"convert_hf_to_gguf.py not found at {convert_script_path}. Run cloning_tools() first.")

        command = [
            "python3",
            convert_script_path,
            source_dir,
            "--outfile", target_gguf_path_f16,
            "--outtype", "f16"
        ]
        try:
            print(f"Converting {source_dir} to GGUF (f16) format...")
            subprocess.run(command, check=True)
            print(f"Successfully converted model to {target_gguf_path_f16}")
        except subprocess.CalledProcessError as e:
            print(f"Error during conversion to GGUF: {e}")
            return
        except FileNotFoundError:
            print(f"Error: convert_hf_to_gguf.py not found. Ensure cloning_tools() has run.")
            return

        quantize_tool_path = os.path.join("./Library", "llama.cpp", "build", "bin", "llama-quantize")
        if not os.path.exists(quantize_tool_path):
            quantize_tool_path = os.path.join("./Library", "llama.cpp", "build", "llama-quantize")
            if not os.path.exists(quantize_tool_path):
                raise FileNotFoundError("llama-quantize executable not found.  Build it using build_llama_quantize().")

        target_gguf_path_quantized = os.path.join(target_dir, f"{model_name}.gguf")
        quantize_command = [
            quantize_tool_path,
            target_gguf_path_f16,
            target_gguf_path_quantized,
            "Q4_K_M"
        ]

        try:
            print(f"Quantizing {target_gguf_path_f16} to {target_gguf_path_quantized}...")
            subprocess.run(quantize_command, check=True)
            print(f"Successfully quantized model to {target_gguf_path_quantized}")
            os.remove(target_gguf_path_f16)
            print(f"Temporary file {target_gguf_path_f16} removed.")
        except subprocess.CalledProcessError as e:
            print(f"Error during quantization: {e}")
        except FileNotFoundError as e:
            print(f"Error: {e}. Ensure you have built llama-quantize.")

    @staticmethod
    def prepare_llm_model(repo_id: str, model_name: str, revision: str = None):
        """Downloads, converts, and quantizes an LLM with LLaVA-specific handling."""
        source_dir = os.path.join("./Model", model_name)
        target_dir = os.path.join("./Model/ModelCompiledRuntime")
        
        # Download original model
        AIModelPreRuntimeManager.download_model(repo_id, source_dir, revision=revision)
        
        # Special handling for LLaVA models
        if "llava" in model_name.lower():
            return AIModelPreRuntimeManager.convert_llava_model(
                source_dir, 
                target_dir,  # Now properly passed
                model_name
            )
        else:
            # Standard conversion for non-LLaVA models
            AIModelPreRuntimeManager.convert_to_gguf(source_dir, target_dir, model_name)
            return os.path.join(target_dir, f"{model_name}.gguf")

    @staticmethod
    def convert_llava_model(source_dir: str, target_dir: str, model_name: str) -> str:
        """
        Converts an LLaVA model into GGUF format with separate main model and projector components.
        Stores FP16 models in Modelfp16PreRuntime and prepares them for JIT quantization.
        """
        # Define directory paths
        target_dir_fp16 = "./Model/Modelfp16PreRuntime"
        os.makedirs(target_dir_fp16, exist_ok=True)

        llama_cpp_path = os.path.join("./Library", "llama.cpp")
        llava_examples_path = os.path.join(llama_cpp_path, "examples", "llava")

        # Step 1: Split LLaVA model into components using llava_surgery_v2.py
        surgery_script = os.path.join(llava_examples_path, "llava_surgery_v2.py")
        split_command = [
            "python3", surgery_script,
            "-C",  # Compatibility flag for newer models
            "-m", source_dir
        ]
        try:
            print("Splitting LLaVA model into components...")
            subprocess.run(split_command, check=True)
            print("LLaVA model split successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error splitting LLaVA model: {e}")
            raise

        # Step 2: Prepare CLIP model directory
        vit_dir = os.path.join(source_dir, "vit")
        os.makedirs(vit_dir, exist_ok=True)
        try:
            print("Preparing CLIP model for GGUF conversion...")
            # Copy CLIP and projector files
            shutil.copy(os.path.join(source_dir, "llava.clip"), 
                        os.path.join(vit_dir, "pytorch_model.bin"))
            shutil.copy(os.path.join(source_dir, "llava.projector"), 
                        os.path.join(vit_dir, "llava.projector"))
            
            # Download CLIP config
            config_url = "https://huggingface.co/cmp-nct/llava-1.6-gguf/raw/main/config_vit.json"
            subprocess.run(["curl", "-s", "-q", config_url, 
                            "-o", os.path.join(vit_dir, "config.json")], check=True)
            print("CLIP model prepared successfully.")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"Error preparing CLIP model: {e}")
            raise

        # Step 3: Convert main LLM to GGUF (FP16)
        llm_dir = os.path.join(source_dir, "language_model")  # Adjust based on surgery output
        convert_script = os.path.join(llama_cpp_path, "examples", "convert_legacy_llama.py")
        main_fp16_path = os.path.join(target_dir_fp16, f"{model_name}-main.gguf")
        main_command = [
            "python3", convert_script,
            llm_dir,
            "--outfile", main_fp16_path,
            "--outtype", "f16",
            "--skip-unknown"  # Required for newer architectures
        ]
        try:
            print("Converting main LLM to GGUF (FP16)...")
            subprocess.run(main_command, check=True)
            print(f"Main LLM converted to GGUF: {main_fp16_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting main LLM: {e}")
            raise

        # Step 4: Convert image encoder (projector) to GGUF (FP16)
        projector_script = os.path.join(llava_examples_path, "convert_image_encoder_to_gguf.py")
        projector_fp16_path = os.path.join(target_dir_fp16, f"{model_name}-mmproj.gguf")
        projector_command = [
            "python3", projector_script,
            "-m", vit_dir,
            "--llava-projector", os.path.join(vit_dir, "llava.projector"),
            "--output-dir", target_dir_fp16,
            "--clip-model-is-vision"
        ]
        try:
            print("Converting vision projector to GGUF (FP16)...")
            subprocess.run(projector_command, check=True)
            print(f"Vision projector converted to GGUF: {projector_fp16_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting vision projector: {e}")
            raise

        # Return path to FP16 main model (used for JIT quantization later)
        return main_fp16_path

    @staticmethod
    def prepare_embedding_model(repo_id: str, model_name: str, revision: str = None):
        """Downloads the embedding model (no conversion needed)."""
        source_dir = os.path.join("./Model", model_name)
        AIModelPreRuntimeManager.download_model(repo_id, source_dir, revision)
        gguf_files = glob.glob(os.path.join(source_dir, "*.gguf"))
        if gguf_files:
            return gguf_files[0]
        else:
            print(f"Warning: No .gguf file found in {source_dir}.")
            return None

    @staticmethod
    def prepare_stable_diffusion_model(repo_id: str, model_name: str, revision: str = None):
        """Downloads the Stable Diffusion model (placeholder)."""
        source_dir = os.path.join("./Model", model_name)
        AIModelPreRuntimeManager.download_model(repo_id, source_dir, revision=revision)
        print("Stable Diffusion model downloaded. Conversion is not yet implemented.")
        return source_dir
    
    @staticmethod
    def jit_quantize(model_path: str, quant_type: str = "Q4_K_M") -> str:
        """Performs JIT quantization and returns path to quantized model"""
        target_dir_runtime = "./Model/ModelCompiledRuntime"
        base_name = os.path.basename(model_path).replace("-main.gguf", "").replace("-mmproj.gguf", "")
        target_path = os.path.join(target_dir_runtime, f"{base_name}.gguf")
        
        if not os.path.exists(target_path):
            quantize_tool = "./Library/llama.cpp/build/bin/llama-quantize"
            subprocess.run([
                quantize_tool,
                model_path,
                target_path,
                quant_type
            ], check=True)
        
        return target_path

    # In model loading code
    def load_quantized_model(self, model_name: str):
        fp16_path = os.path.join("./Model/Modelfp16PreRuntime", f"{model_name}-main.gguf")
        quant_type = self.determine_optimal_quantization()
        quantized_path = self.jit_quantize(fp16_path, quant_type)
        return Llama(model_path=quantized_path)

    def determine_optimal_quantization(self) -> str:
        """Determines best quantization based on hardware"""
        # Add actual hardware detection logic here
        return "Q4_K_M"  # Default quantization


class AIRuntimeManager:
    def __init__(self, llm_instance, database_manager):
        self.llm = llm_instance
        self.current_task = None
        self.task_queue = []
        self.backbrain_tasks = []
        self.lock = Lock()
        self.last_task_info = {}
        self.start_time = None
        self.fuzzy_threshold = 0.69
        self.database_manager = database_manager
        self.gguf_parser: Optional[GGUFParser] = None  # Store GGUF parser here
        self.chat_formatter = ChatFormatter(self.gguf_parser)
        self.partition_context = None
        self.is_llm_running = False  # Flag to indicate LLM usage
        self._model_lock = threading.Lock()  # A lock to serialize access to the LLM
        self._stop_event = threading.Event()
        self.llm_thread = None  # Keep track of running LLM threads.
        # Start the scheduler thread
        self.scheduler_thread = Thread(target=self.scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

        self.json_interpreter = JSONInterpreter(self.invoke_llm)

        # Start the branch_predictor thread
        self.branch_predictor_thread = Thread(target=self.branch_predictor)
        self.branch_predictor_thread.daemon = True
        self.branch_predictor_thread.start()

        # Start reporting thread after other threads
        self.start_reporting_thread()

        # Load the task queue from the database during initialization
        self.task_queue, self.backbrain_tasks = self.database_manager.load_task_queue()
        self.last_queue_save_time = time.time()
        self.queue_save_interval = 60  # Save the queue every 60 seconds (adjust as needed)

    def initialize_gguf_parser(self):
        """Initializes the GGUF parser."""
        if LLM_MODEL_PATH:  # Only initialize if a path is set
            self.gguf_parser = GGUFParser(LLM_MODEL_PATH)
            # Example usage: print the metadata
            if self.gguf_parser:
                metadata = self.gguf_parser.get_metadata()
                print(OutputFormatter.color_prefix("GGUF Metadata:", "Internal"))
                for key, value in metadata.items():
                    print(OutputFormatter.color_prefix(f"  {key}: {value}", "Internal"))

    def process_json_response(self, json_response):
        """Processes a JSON response, attempting to repair it if invalid."""
        repaired_json = self.json_interpreter.repair_json(json_response)

        if repaired_json:
            try:
                data = json.loads(repaired_json)
                print(OutputFormatter.color_prefix(f"Successfully processed JSON data: {data}", "Internal"))
                return data
            except json.JSONDecodeError:
                print(OutputFormatter.color_prefix("Critical Error: Failed to parse JSON after repair.", "Internal"))
                return None  # Or raise, or return a default value
        else:
            print(OutputFormatter.color_prefix("Failed to repair JSON.", "Internal"))
            return None

    def force_unload_model(self):
        """Forces the unloading of the LLM and exits."""
        with self._model_lock:  # Ensure exclusive access to model operations
            print(OutputFormatter.color_prefix("Forcefully unloading the model...", "Internal"))
            self._stop_event.set()  # Signal the thread to stop.
            if self.llm_thread and self.llm_thread.is_alive():
                try:
                    # Windows-specific: Use TerminateThread
                    if platform.system() == "Windows":
                        import ctypes
                        handle = self.llm_thread.native_id
                        ctypes.windll.kernel32.TerminateThread(handle, 0)
                        print(OutputFormatter.color_prefix("LLM thread terminated (Windows).", "Internal"))
                    else:  # POSIX-compliant systems: Use pthread_kill with SIGKILL
                        import signal
                        import ctypes
                        pthread_kill = ctypes.CDLL("libc.so.6").pthread_kill  # or similar
                        pthread_kill(self.llm_thread.native_id, signal.SIGKILL)

                        print(OutputFormatter.color_prefix("LLM thread terminated (POSIX).", "Internal"))
                except Exception as e:
                    print(OutputFormatter.color_prefix(f"Error terminating LLM thread: {e}", "Internal"))

            self.llm = None  # Release the LLM object
            self.is_llm_running = False
            global embedding_model, vector_store
            embedding_model = None
            vector_store = None
            print(OutputFormatter.color_prefix("Model unloaded.", "Internal"))
            database_manager.close()  # Close the database connection
            os._exit(1)  # Exit immediately, by passing regular shutdown processes

    def unload_model(self):
        """Unloads the LLM from memory, safely interrupting any running inference."""
        with self._model_lock:
            print(OutputFormatter.color_prefix("Unloading the model...", "Internal"))
            self._stop_event.set()  # Signal any running LLM thread to stop

            if self.llm_thread and self.llm_thread.is_alive():
                try:
                    # OS-specific thread termination
                    if platform.system() == "Windows":
                        handle = self.llm_thread.native_id
                        ctypes.windll.kernel32.TerminateThread(ctypes.c_void_p(handle), 0)
                        print(OutputFormatter.color_prefix("LLM thread terminated (Windows).", "Internal"))

                    else:  # POSIX-compliant
                        res = ctypes.pythonapi.pthread_kill(ctypes.c_ulong(self.llm_thread.native_id), signal.SIGKILL)
                        if res == 0:
                            print(OutputFormatter.color_prefix("LLM thread terminated (POSIX).", "Internal"))
                        elif res != 1:  # ESRCH (No such process) is acceptable
                            print(OutputFormatter.color_prefix("Failed to terminate LLM thread.", "Internal"))

                except Exception as e:
                    print(OutputFormatter.color_prefix(f"Error terminating LLM thread: {e}", "Internal"))

                self.llm_thread.join()  # Wait for the thread to actually finish

            self.llm = None
            self.is_llm_running = False
            global embedding_model, vector_store  # Also unload embeddings
            embedding_model = None
            vector_store = None
            self._stop_event.clear()  # Reset for the next LLM run
            print(OutputFormatter.color_prefix("Model unloaded.", "Internal"))

    def add_task(self, task, args=(), priority=0):  # Modified
        """Adds a task to the appropriate queue based on priority."""
        with self.lock:
            task_item = (task, args)  # Combine task and args into a single tuple

            if priority == 0:
                self.task_queue.append((task_item, priority))
            elif priority == 1:
                self.task_queue.append((task_item, priority))
            elif priority == 2:
                self.task_queue.append((task_item, priority))
            elif priority == 3:
                self.backbrain_tasks.append((task_item, priority))
            elif priority == 4:
                self.task_queue.append((task_item, priority))
            elif priority == 99:
                if not hasattr(self, 'mesh_network_tasks'):
                    self.mesh_network_tasks = []
                self.mesh_network_tasks.append((task_item, priority))
            else:
                raise ValueError("Invalid priority level.")

            if time.time() - self.last_queue_save_time > self.queue_save_interval:
                self.database_manager.save_task_queue(self.task_queue, self.backbrain_tasks,
                                                      self.mesh_network_tasks if hasattr(self,
                                                                                         'mesh_network_tasks') else [])
                self.last_queue_save_time = time.time()

    def get_next_task(self):
        with self.lock:
            """Gets the next task from the highest priority queue that is not empty."""
            print(
                OutputFormatter.color_prefix(f"get_next_task: task_queue = {self.task_queue}", "Internal"))  # Debugging
            print(OutputFormatter.color_prefix(f"get_next_task: backbrain_tasks = {self.backbrain_tasks}",
                                               "Internal"))  # Debugging
            if hasattr(self, 'mesh_network_tasks'):
                print(OutputFormatter.color_prefix(f"get_next_task: mesh_network_tasks = {self.mesh_network_tasks}",
                                                   "Internal"))  # Debugging

            if self.task_queue:
                return self.task_queue.pop(0)  # FIFO
            elif self.backbrain_tasks:
                return self.backbrain_tasks.pop(0)
            elif hasattr(self, 'mesh_network_tasks') and self.mesh_network_tasks:  # check priority level 99 tasks
                return self.mesh_network_tasks.pop(0)
            else:
                return None

            if time.time() - self.last_queue_save_time > self.queue_save_interval:
                self.database_manager.save_task_queue(self.task_queue, self.backbrain_tasks,
                                                      self.mesh_network_tasks if hasattr(self,
                                                                                         'mesh_network_tasks') else [])
                self.last_queue_save_time = time.time()

    def cached_inference(self, prompt, slot, context_type):
        """Checks if a similar prompt exists in the database using fuzzy matching."""

        db_cursor = self.database_manager.db_cursor
        db_cursor.execute("""
            SELECT interaction_history.response
            FROM interaction_history
            WHERE interaction_history.slot = ? AND interaction_history.context_type = ?
        """, (slot, context_type))
        cached_results = db_cursor.fetchall()

        best_match = None
        best_score = 0

        for (cached_response,) in cached_results:
            score = fuzz.ratio(prompt, cached_response) / 100.0
            if score > best_score and score >= self.fuzzy_threshold:
                best_match = cached_response
                best_score = score

        if best_match:
            print(
                OutputFormatter.color_prefix(f"Found cached inference with score: {best_score}", "BackbrainController"))
            return best_match
        else:
            return None

    def add_to_cache(self, prompt, response, context_type, slot):
        """Adds a prompt-response pair to the chat_history table with context_type 'cached'."""
        try:
            self.database_manager.db_writer.schedule_write(
                "INSERT INTO interaction_history (slot, role, message, response, context_type) VALUES (?, ?, ?, ?, ?)",
                (slot, "User", prompt, response, context_type)
            )
        except sqlite3.IntegrityError:
            print(OutputFormatter.color_prefix("Prompt already exists in cache. Skipping insertion.",
                                               "BackbrainController"))
        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error adding to cache: {e}", "BackbrainController"))

    def scheduler(self):
        """Scheduler loop (modified to use process_json_response)."""
        while not self._stop_event.is_set():
            try:  # Added broad exception handling
                task = self.get_next_task()
                if task:
                    task_item, priority = task  # Unpack the outer tuple: ((task, args), priority)
                    task_callable, task_args = task_item  # Unpack the inner tuple: (task, args)

                    self.start_time = time.time()
                    task_name = task_callable.__name__ if hasattr(task_callable, '__name__') else str(
                        task_callable)  # Gets the task name.

                    print(OutputFormatter.color_prefix(
                        f"Starting task: {task_name} with priority {priority}",
                        "BackbrainController",
                        self.start_time
                    ))

                    try:  # Execute the task.
                        if task_callable == self.generate_response:
                            timeout = 60 if priority == 0 else None
                            result = None

                            def run_llm_task():
                                nonlocal result
                                try:
                                    print(OutputFormatter.color_prefix(
                                        f"run_llm_task: _stop_event.is_set(): {self._stop_event.is_set()}",
                                        "Internal"))  # DEBUG
                                    if self._stop_event.is_set():
                                        print(OutputFormatter.color_prefix("Stop event set. Skipping LLM invocation.",
                                                                           "BackbrainController"))
                                        return

                                    # if len(task_args) == 3: # Removed this check
                                    #     result = task_callable(*task_args)
                                    # else:
                                    #     result = task_callable(task_args[0], task_args[1])
                                    result = task_callable(*task_args)  # Always unpack args.
                                except Exception as e:
                                    print(
                                        OutputFormatter.color_prefix(f"Error in LLM task: {e}", "BackbrainController"))
                                    traceback.print_exc()  # Print full traceback

                            with self._model_lock:
                                print(OutputFormatter.color_prefix(
                                    f"scheduler: Acquiring _model_lock for generate_response", "Internal"))
                                if self.llm is None:
                                    initialize_models()
                                    self.llm = ai_runtime_manager.llm  # Corrected line
                                    self.partition_context.vector_store = vector_store
                                print(OutputFormatter.color_prefix(
                                    f"scheduler: Acquired _model_lock for generate_response", "Internal"))
                                self._stop_event.clear()
                                print(OutputFormatter.color_prefix(f"scheduler: Cleared _stop_event", "Internal"))

                            if priority != 4:  # Handle cached inference for non-branch-prediction tasks
                                user_input, slot, *other_args = task_args
                                context_type = "CoT" if priority == 3 else "main"

                                cached_response = self.cached_inference(user_input, slot, context_type)
                                if cached_response:
                                    print(OutputFormatter.color_prefix(f"Using cached response for slot {slot}",
                                                                       "BackbrainController"))
                                    if priority != 0:
                                        self.partition_context.add_context(slot, cached_response, context_type)
                                    elif priority == 0:
                                        result = cached_response  # Assign to result, for the return
                                        continue  # Skips the rest

                            if priority != 4:
                                self.llm_thread = threading.Thread(target=run_llm_task)
                                self.llm_thread.start()
                            else:  # No thread needed
                                run_llm_task()

                            if timeout is not None:
                                self.llm_thread.join(timeout)
                                if self.llm_thread.is_alive():
                                    print(OutputFormatter.color_prefix(
                                        f"Task {task_callable.__name__} timed out after {timeout} seconds.",
                                        "BackbrainController",
                                        time.time() - self.start_time
                                    ))
                                    self.unload_model()
                                    return  # Exit the scheduler loop on timeout
                            elif priority != 4:
                                self.llm_thread.join()

                        elif task_callable == self.process_branch_prediction_slot:  # Modified section
                            slot, chat_history = task_args  # Unpack.

                            decision_tree_prompt = self.create_decision_tree_prompt(chat_history)

                            with self._model_lock:  # Added lock
                                print(OutputFormatter.color_prefix(
                                    f"scheduler: Acquiring _model_lock for process_branch_prediction_slot",
                                    "Internal"))  # DEBUG
                                decision_tree_text = self.invoke_llm(decision_tree_prompt,
                                                                     caller="process_branch_prediction_slot")
                                print(OutputFormatter.color_prefix(
                                    f"scheduler: Acquired _model_lock for process_branch_prediction_slot",
                                    "Internal"))  # DEBUG

                            json_tree_prompt = self.create_json_tree_prompt(decision_tree_text)
                            with self._model_lock:  # Added lock
                                print(OutputFormatter.color_prefix(
                                    f"scheduler: Acquiring _model_lock for process_branch_prediction_slot (2nd LLM call)",
                                    "Internal"))  # DEBUG
                                json_tree_response = self.invoke_llm(json_tree_prompt,
                                                                     caller="process_branch_prediction_slot")
                                print(OutputFormatter.color_prefix(
                                    f"scheduler: Acquired _model_lock for process_branch_prediction_slot (2nd LLM call)",
                                    "Internal"))  # DEBUG

                            # --- Use process_json_response ---
                            decision_tree_json = self.process_json_response(json_tree_response)
                            # --- End of modification ---

                            if decision_tree_json:  # Check for None
                                potential_inputs = self.extract_potential_inputs(decision_tree_json)

                                for user_input in potential_inputs:
                                    print(OutputFormatter.color_prefix(
                                        f"Scheduling generate_response for predicted input: {user_input}",
                                        "branch_predictor"))
                                    prefixed_input = f"branch_predictor: {user_input}"
                                    self.add_task((self.generate_response, (prefixed_input, slot)), 4)
                            else:
                                print(OutputFormatter.color_prefix("Decision tree JSON processing failed.",
                                                                   "branch_predictor"))


                        else:
                            result = task_callable(*task_args)

                    except Exception as e:  # Exception Handling
                        print(OutputFormatter.color_prefix(
                            f"Task {task_callable.__name__} raised an exception: {e}",
                            "BackbrainController",
                            time.time() - self.start_time
                        ))
                        traceback.print_exc()  # Print the full traceback

                    else:  # Execute 'else' block if no exception occurred.
                        elapsed_time = time.time() - self.start_time
                        if task_callable == self.generate_response:
                            if priority == 0 and elapsed_time < 58:
                                pass
                            elif priority != 4:
                                user_input, slot, *_ = task_args
                                context_type = "CoT" if priority == 3 else "main"

                                # --- JSON Processing for decision and evaluation ---
                                if context_type == "CoT":
                                    if "Decision:" in result:  # Check for decision prompt
                                        result = self.process_json_response(result)  # Process JSON
                                    elif "Evaluation:" in result:  # Check for evaluation prompt
                                        result = self.process_json_response(result)  # Process JSON

                                # --- End of JSON Processing ---
                                if result is not None:  # Proceed if not None
                                    self.add_to_cache(user_input,
                                                      result if isinstance(result, str) else json.dumps(result),
                                                      context_type, slot)
                                    self.partition_context.add_context(slot, result if isinstance(result,
                                                                                                  str) else json.dumps(
                                        result), "main" if priority == 0 else "CoT")
                                    asyncio.run_coroutine_threadsafe(
                                        self.partition_context.async_embed_and_store(
                                            result if isinstance(result, str) else json.dumps(result), slot,
                                            "main" if priority == 0 else "CoT"),
                                        loop
                                    )

                        self.last_task_info = {
                            "task": task_callable,
                            "args": task_args,
                            "result": result,
                            "elapsed_time": elapsed_time,
                        }
                        print(OutputFormatter.color_prefix(
                            f"Finished task: {task_callable.__name__} in {elapsed_time:.2f} seconds",
                            "BackbrainController",
                            time.time() - self.start_time
                        ))


                    finally:
                        self.current_task = None  # Always clear the current task
                        if self.llm_thread:
                            self._stop_event.clear()
                        if task_callable == self.generate_response and priority == 0:
                            return result

                else:
                    time.sleep(0.095)
                    if time.time() - self.last_queue_save_time > self.queue_save_interval:
                        with self.lock:
                            self.database_manager.save_task_queue(self.task_queue, self.backbrain_tasks,
                                                                  self.mesh_network_tasks if hasattr(self,
                                                                                                     'mesh_network_tasks') else [])
                            self.last_queue_save_time = time.time()

            except Exception as e:  # Added broad exception handling.
                print(OutputFormatter.color_prefix(f"FATAL ERROR IN SCHEDULER: {e}", "BackbrainController"))
                traceback.print_exc()  # Print full traceback

    def invoke_llm(self, prompt, caller="Unknown Caller"):
        """
        Invokes the LLM, handling potential model unloading and checking for
        existing invocations. Includes debug output and error handling.
        """
        with self._model_lock:  # Acquire the lock *before* checking/setting is_llm_running
            print(OutputFormatter.color_prefix(f"invoke_llm called by {caller}. is_llm_running: {self.is_llm_running}",
                                               "Internal"))
            if self.is_llm_running:
                print(OutputFormatter.color_prefix(
                    "LLM invocation already in progress. Skipping this invocation.", "Internal"
                ))
                return "LLM busy."  # Or raise an exception

            self.is_llm_running = True
            print(OutputFormatter.color_prefix(f"invoke_llm: Set is_llm_running to True", "Internal"))

            try:
                start_time = time.time()
                prompt_tokens = len(TOKENIZER.encode(prompt))
                print(OutputFormatter.color_prefix(f"Invoking LLM with prompt (called by {caller}):\n{prompt}",
                                                   "Internal"))

                print(OutputFormatter.color_prefix(f"EXACT PROMPT: {repr(prompt)}", "Internal"))  # ADDED THIS

                if prompt_tokens > int(CTX_WINDOW_LLM * 0.75):
                    print(OutputFormatter.color_prefix(f"Prompt exceeds 75% of context window. Truncating...",
                                                       "BackbrainController"))
                    truncated_prompt = TOKENIZER.decode(TOKENIZER.encode(prompt)[:int(CTX_WINDOW_LLM * 0.75)])
                    if truncated_prompt[-1] not in [".", "?", "!"]:
                        last_period_index = truncated_prompt.rfind(".")
                        last_question_index = truncated_prompt.rfind("?")
                        last_exclamation_index = truncated_prompt.rfind("!")
                        last_punctuation_index = max(last_period_index, last_question_index, last_exclamation_index)
                        if last_punctuation_index != -1:
                            truncated_prompt = truncated_prompt[:last_punctuation_index + 1]
                    print(OutputFormatter.color_prefix("Truncated prompt being used...", "BackbrainController"))
                    response = self.llm(truncated_prompt, stop=[])  # ADDED stop=[]
                else:
                    response = self.llm(prompt, stop=[])  # ADDED stop=[]

                print(OutputFormatter.color_prefix(f"LLM response (called by {caller}):\n{response}", "Internal"))
                if response and 'choices' in response and len(response['choices']) > 0:
                    return response['choices'][0]['text'].strip()
                return ""

            except Exception as e:
                print(OutputFormatter.color_prefix(f"Error during LLM invocation: {e}", "Internal"))
                traceback.print_exc()
                return "Error during LLM invocation."

            finally:
                self.is_llm_running = False
                print(OutputFormatter.color_prefix(f"invoke_llm: Set is_llm_running to False in finally block",
                                                   "Internal"))

    def report_queue_status(self):  # Modified to report the meshNetworkProcessingIO Queue
        """Reports the queue status (length and contents) every 10 seconds."""
        while True:
            with self.lock:
                task_queue_length = len(self.task_queue)
                backbrain_tasks_length = len(self.backbrain_tasks)
                mesh_network_tasks_length = len(self.mesh_network_tasks) if hasattr(self, 'mesh_network_tasks') else 0

                # Correctly handle the task item structure
                task_queue_contents = [
                    ((task.__name__ if callable(task) else str(task)), args) for (task, args), _ in self.task_queue
                ]
                backbrain_tasks_contents = [
                    ((task.__name__ if callable(task) else str(task)), args) for (task, args), _ in self.backbrain_tasks
                ]
                mesh_network_tasks_contents = [
                    ((task.__name__ if callable(task) else str(task)), args) for (task, args), _ in
                    self.mesh_network_tasks
                ] if hasattr(self, 'mesh_network_tasks') else []

            print(OutputFormatter.color_prefix(
                f"Task Queue Length: {task_queue_length} | Contents: {task_queue_contents}",
                "BackbrainController"
            ))
            print(OutputFormatter.color_prefix(
                f"Backbrain Tasks Length: {backbrain_tasks_length} | Contents: {backbrain_tasks_contents}",
                "BackbrainController"
            ))
            print(OutputFormatter.color_prefix(  # queue report
                f"Mesh Network Tasks Length: {mesh_network_tasks_length} | Contents: {mesh_network_tasks_contents}",
                "BackbrainController"
            ))

            time.sleep(10)

    def start_reporting_thread(self):
        """Starts the thread that reports the queue status."""
        reporting_thread = Thread(target=self.report_queue_status)
        reporting_thread.daemon = True
        reporting_thread.start()

    def run_with_timeout(self, func, args, timeout):
        """This function is no longer directly used for timed execution."""
        pass  # Remove the content, as it's now handled in scheduler.

    def branch_predictor(self):
        """
        Analyzes chat history, predicts likely user inputs, and schedules LLM invocations for decision tree processing.
        """
        time.sleep(60)
        while True:
            try:
                for slot in range(5):
                    print(OutputFormatter.color_prefix(f"branch_predictor analyzing slot {slot}...",
                                                       "BackbrainController"))
                    chat_history = self.database_manager.get_chat_history(slot)

                    if not chat_history:
                        continue

                    # Schedule decision tree generation as a task
                    self.add_task((self.process_branch_prediction_slot, (slot, chat_history)), 4)

            except Exception as e:
                print(OutputFormatter.color_prefix(f"Error in branch_predictor: {e}", "BackbrainController"))

            time.sleep(5)

    def process_branch_prediction_slot(self, slot, chat_history):
        """
        Generates and processes the decision tree for a given slot's chat history.
        This is now a separate function to be executed as a task.
        """
        decision_tree_prompt = self.create_decision_tree_prompt(chat_history)

        # Invoke LLM within the task, ensuring sequential execution
        while self.is_llm_running:
            time.sleep(0.1)
        self.is_llm_running = True
        try:
            decision_tree_text = self.invoke_llm(decision_tree_prompt, caller="process_branch_prediction_slot")

            json_tree_prompt = self.create_json_tree_prompt(decision_tree_text)

            # Invoke LLM again, ensuring sequential execution
            while self.is_llm_running:
                time.sleep(0.1)
            json_tree_response = self.invoke_llm(json_tree_prompt, caller="process_branch_prediction_slot")
            decision_tree_json = self.parse_decision_tree_json(json_tree_response)

            potential_inputs = self.extract_potential_inputs(decision_tree_json)

            for user_input in potential_inputs:
                print(OutputFormatter.color_prefix(f"Scheduling generate_response for predicted input: {user_input}",
                                                   "branch_predictor"))
                prefixed_input = f"branch_predictor: {user_input}"
                # Add generate_response task with the predicted input (still priority 4)
                self.add_task((self.generate_response, (prefixed_input, slot)), 4)
        finally:
            self.is_llm_running = False

    def create_decision_tree_prompt(self, chat_history):
        """Creates a prompt for generating a decision tree based on chat history."""
        history_text = "\n".join([f"{msg[0]}: {msg[1]}" for msg in chat_history])
        prompt = f"""
        Analyze the following chat history and create a decision tree to predict likely user inputs:
        {history_text}

        The decision tree should outline key decision points and potential user actions or questions.
        """
        return prompt

    def create_json_tree_prompt(self, decision_tree_text):
        """Creates a prompt for converting a decision tree to JSON format."""
        prompt = f"""
        Convert the following decision tree to JSON, adhering to the specified format.
        Decision Tree:
        {decision_tree_text}

        The JSON *must* be complete and follow this format:
        ```json
        {{
            "input": "User input text",
            "initial_response": "Initial response generated by the system",
            "nodes": [
                {{
                    "node_id": "unique identifier for the node",
                    "node_type": "question, action step, conclusion, or reflection",
                    "content": "Text content of the node",
                    "options": [
                      {{
                          "option_id": "unique identifier for the option",
                          "next_node_id": "node_id of the next node if this option is chosen",
                          "option_text": "Description of the option"
                      }}
                    ]
                }}
            ],
            "edges": [
                {{
                    "from_node_id": "node_id of the source node",
                    "to_node_id": "node_id of the destination node",
                    "condition": "Optional condition for taking this edge"
                }}
            ]
        }}
        ```
        Do not stop generating until you are sure the JSON is complete and syntactically correct as defined in the format.
        Respond with JSON, and only JSON, strictly adhering to the above format.
        """
        return prompt

    def parse_decision_tree_json(self, json_tree_response):
        """Parses the decision tree JSON, using the JSONInterpreter."""
        # Use the process_json_response method
        return self.process_json_response(json_tree_response)

    def extract_potential_inputs(self, decision_tree_json):
        """Extracts potential user inputs from the decision tree JSON."""
        potential_inputs = []
        if decision_tree_json:
            nodes = decision_tree_json.get("nodes", [])
            for node in nodes:
                if node["node_type"] == "question":
                    potential_inputs.append(node["content"])
        return potential_inputs

    def _prepare_prompt(self, user_input, slot, is_v1_completions=False):
        """Prepares the prompt, using the GGUF-aware ChatFormatter."""
        global assistantName  # Ensure assistantName is accessible

        decoded_initial_instructions = base64.b64decode(encoded_instructions.strip()).decode("utf-8")
        decoded_initial_instructions = decoded_initial_instructions.replace("${assistantName}", assistantName)

        context_messages = [{"role": "system", "content": decoded_initial_instructions}]

        if is_v1_completions:
            if isinstance(user_input, str):
                try:
                    messages = json.loads(user_input)
                except json.JSONDecodeError:
                    print(OutputFormatter.color_prefix("Invalid JSON in /v1/completions request.", "Internal"))
                    return None
            else:
                messages = user_input
            context_messages.extend(messages)

        else:
            self.partition_context.add_context(slot, f"User: {user_input}", "main")
            asyncio.run_coroutine_threadsafe(
                self.partition_context.async_embed_and_store(f"User: {user_input}", slot, "main"), loop)
            main_history = self.database_manager.fetch_chat_history(slot)
            context = self.partition_context.get_context(slot, "main")

            if context:
                for entry in context:
                    context_messages.append({"role": "user", "content": entry})
            if main_history:
                for entry in main_history:
                    context_messages.append(entry)

        print(OutputFormatter.color_prefix(f"context_messages: {context_messages}", "Internal"))  # DEBUG

        # --- Use the self.chat_formatter ---
        prompt = self.chat_formatter.create_prompt(messages=context_messages, add_generation_prompt=True)
        # --- End of modification ---
        return prompt

    def generate_response(self, user_input, slot, stream=False, decoded_initial_instructions=None):  # Modified
        """Generates a response, using CoT if necessary, and processing JSON."""
        global assistantName

        start_time = time.time()
        is_v1_completions = isinstance(user_input, str) and user_input.startswith("{")

        prompt = self._prepare_prompt(user_input, slot, is_v1_completions)
        if prompt is None:
            return "Error: Invalid input format for /v1/completions."

        prompt_tokens = len(TOKENIZER.encode(prompt))

        if is_v1_completions:
            print(OutputFormatter.color_prefix("Direct query (/v1/completions). Generating direct response...",
                                               "Internal", time.time() - start_time, progress=10, slot=slot))
            direct_response = self.invoke_llm(prompt)  # Corrected call
            asyncio.run_coroutine_threadsafe(self.database_manager.async_db_write(slot, user_input, direct_response),
                                             loop)  # Async write
            end_time = time.time()
            generation_time = end_time - start_time
            print(OutputFormatter.color_prefix(direct_response, "Adelaide", generation_time, token_count=prompt_tokens,
                                               slot=slot))
            return direct_response

        print(OutputFormatter.color_prefix("Deciding whether to engage in deep thinking...", "Internal",
                                           time.time() - start_time, progress=0, token_count=prompt_tokens, slot=slot))
        decision_prompt = f"""
        Analyze the input and decide if it requires in-depth processing or a simple response.
        Input: "{user_input}"

        Provide a JSON response in the following format:
        ```json
        {{
            "decision": "<yes or no>",
            "reasoning": "<A very short one-paragraph summary of why this decision was made.>"
        }}
        ```
        Make sure the JSON you generate is valid and adheres to the formatting I have given. Do not stop generating before you can confirm it is valid.

        Respond with JSON, and only JSON, strictly adhering to the above format.
        """

        decision_prompt_tokens = len(TOKENIZER.encode(decision_prompt))
        print(OutputFormatter.color_prefix("Processing Decision Prompt", "Internal", time.time() - start_time,
                                           token_count=decision_prompt_tokens, progress=1, slot=slot))

        decision_response = self.invoke_llm(decision_prompt)  # Corrected call
        asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(decision_response, slot, "CoT"),
                                         loop)  # Moved Here
        decision_json = self.process_json_response(decision_response)

        deep_thinking_required = False  # Default value
        reasoning_summary = ""
        if decision_json:
            deep_thinking_required = decision_json.get("decision", "no").lower() == "yes"
            reasoning_summary = decision_json.get("reasoning", "")
        # --- End of JSON processing ---

        print(OutputFormatter.color_prefix(
            f"Decision: {'Deep thinking required' if deep_thinking_required else 'Simple response sufficient'}",
            "Internal", time.time() - start_time, progress=5, slot=slot))
        print(OutputFormatter.color_prefix(f"Reasoning: {reasoning_summary}", "Internal", time.time() - start_time,
                                           progress=5, slot=slot))

        if not deep_thinking_required:
            print(OutputFormatter.color_prefix("Simple query detected. Generating a direct response...", "Internal",
                                               time.time() - start_time, progress=10, slot=slot))
            relevant_context = self.partition_context.get_relevant_chunks(user_input, slot, k=5, requester_type="main")
            if relevant_context:
                retrieved_context_text = "\n".join([item[0] for item in relevant_context])
                # context_messages = [{"role": "system", "content": decoded_initial_instructions}] # REMOVED
                context_messages = [{"role": "system", "content": decoded_initial_instructions}]

                if relevant_context:
                    for entry in relevant_context:
                        context_messages.append({"role": "user", "content": entry[0]})

                main_history = self.database_manager.fetch_chat_history(slot)
                if main_history:
                    for entry in main_history:
                        context_messages.append(entry)
                context_messages.append({"role": "user", "content": user_input})

                prompt = self.chat_formatter.create_prompt(messages=context_messages, add_generation_prompt=True)

            else:  # ADDED ELSE
                prompt = self._prepare_prompt(user_input, slot)  # ADDED

            direct_response = self.invoke_llm(prompt)  # Corrected call

            # --- Add to context and embed NOW (for simple responses) ---
            self.partition_context.add_context(slot, direct_response, "main")
            asyncio.run_coroutine_threadsafe(
                self.partition_context.async_embed_and_store(direct_response, slot, "main"), loop)
            # ---

            asyncio.run_coroutine_threadsafe(self.database_manager.async_db_write(slot, user_input, direct_response),
                                             loop)
            end_time = time.time()
            generation_time = end_time - start_time
            print(OutputFormatter.color_prefix(direct_response, "Adelaide", generation_time, token_count=prompt_tokens,
                                               slot=slot))

            return direct_response

        print(OutputFormatter.color_prefix("Engaging in deep thinking process...", "Internal", time.time() - start_time,
                                           progress=10, slot=slot))
        # --- Add to CoT context only for deep thinking ---
        self.partition_context.add_context(slot, user_input, "CoT")
        asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(user_input, slot, "CoT"), loop)
        # ---

        print(OutputFormatter.color_prefix("Generating initial direct answer...", "Internal", time.time() - start_time,
                                           progress=15, slot=slot))
        initial_response_prompt = f"{prompt}\nProvide a concise initial response."

        initial_response_prompt_tokens = len(TOKENIZER.encode(initial_response_prompt))
        print(OutputFormatter.color_prefix("Processing Initial Response Prompt", "Internal", time.time() - start_time,
                                           token_count=initial_response_prompt_tokens, progress=16, slot=slot))

        initial_response = self.invoke_llm(initial_response_prompt)  # Corrected call
        # --- Add to CoT and embed ---
        asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(initial_response, slot, "CoT"),
                                         loop)
        print(
            OutputFormatter.color_prefix(f"Initial response: {initial_response}", "Internal", time.time() - start_time,
                                         progress=20, slot=slot))

        print(OutputFormatter.color_prefix("Creating a to-do list for in-depth analysis...", "Internal",
                                           time.time() - start_time, progress=25, slot=slot))
        todo_prompt = f"""
        {prompt}\n
        Based on the query '{user_input}', list the steps for in-depth analysis.
        Include search queries for external resources, ending with self-reflection.
        """

        todo_prompt_tokens = len(TOKENIZER.encode(todo_prompt))
        print(OutputFormatter.color_prefix("Processing To-do Prompt", "Internal", time.time() - start_time,
                                           token_count=todo_prompt_tokens, progress=26, slot=slot))

        todo_response = self.invoke_llm(todo_prompt)  # Corrected call
        asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(todo_response, slot, "CoT"), loop)

        print(OutputFormatter.color_prefix(f"To-do list: {todo_response}", "Internal", time.time() - start_time,
                                           progress=30, slot=slot))

        search_queries = re.findall(r"literature_review\(['\"](.*?)['\"]\)", todo_response)
        for query in search_queries:
            LiteratureReviewer.literature_review(query)

        print(OutputFormatter.color_prefix("Creating a decision tree for action planning...", "Internal",
                                           time.time() - start_time, progress=35, slot=slot))
        decision_tree_prompt = f"{prompt}\nGiven the to-do list '{todo_response}', create a decision tree for actions."

        decision_tree_prompt_tokens = len(TOKENIZER.encode(decision_tree_prompt))
        print(OutputFormatter.color_prefix("Processing Decision Tree Prompt", "Internal", time.time() - start_time,
                                           token_count=decision_tree_prompt_tokens, progress=36, slot=slot))

        decision_tree_text = self.invoke_llm(decision_tree_prompt)  # Corrected call
        asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(decision_tree_text, slot, "CoT"),
                                         loop)
        print(OutputFormatter.color_prefix(f"Decision tree (text): {decision_tree_text}", "Internal",
                                           time.time() - start_time, progress=40, slot=slot))

        print(OutputFormatter.color_prefix("Converting decision tree to JSON...", "Internal", time.time() - start_time,
                                           progress=45, slot=slot))
        json_tree_prompt = self.create_json_tree_prompt(decision_tree_text)

        json_tree_prompt_tokens = len(TOKENIZER.encode(json_tree_prompt))
        print(OutputFormatter.color_prefix("Processing JSON Tree Prompt", "Internal", time.time() - start_time,
                                           token_count=json_tree_prompt_tokens, progress=46, slot=slot))

        json_tree_response = self.invoke_llm(json_tree_prompt)  # Corrected call

        # --- Use process_json_response for decision tree ---
        decision_tree_json = self.process_json_response(json_tree_response)
        if decision_tree_json:
            print(OutputFormatter.color_prefix(f"Decision tree (JSON): {decision_tree_json}", "Internal",
                                               time.time() - start_time, progress=55, slot=slot))
            nodes = decision_tree_json.get("nodes", [])
            num_nodes = len(nodes)

            for i, node in enumerate(nodes):
                progress_interval = 55 + (i / num_nodes) * 35
                DecisionTreeProcessor.process_node(node, prompt, start_time, progress_interval, self.partition_context,
                                                   slot)

            print(
                OutputFormatter.color_prefix("Formulating a conclusion based on processed decision tree...", "Internal",
                                             time.time() - start_time, progress=90, slot=slot))
            conclusion_prompt = f"""
            {prompt}\n
            Synthesize a comprehensive conclusion from these insights:\n
            Initial Response: {initial_response}\n
            To-do List: {todo_response}\n
            Decision Tree (text): {decision_tree_text}\n
            Processed Decision Tree Nodes: {self.partition_context.get_context(slot, "CoT")}\n

            Provide a final conclusion based on the entire process.
            """

            conclusion_prompt_tokens = len(TOKENIZER.encode(conclusion_prompt))
            print(OutputFormatter.color_prefix("Processing Conclusion Prompt", "Internal", time.time() - start_time,
                                               token_count=conclusion_prompt_tokens, progress=91, slot=slot))

            conclusion_response = self.invoke_llm(conclusion_prompt)  # Corrected call
            asyncio.run_coroutine_threadsafe(
                self.partition_context.async_embed_and_store(conclusion_response, slot, "CoT"), loop)
            print(OutputFormatter.color_prefix(f"Conclusion (after decision tree processing): {conclusion_response}",
                                               "Internal", time.time() - start_time, progress=92, slot=slot))

        else:
            print(OutputFormatter.color_prefix("Error: Could not parse decision tree JSON after multiple retries.",
                                               "Internal", time.time() - start_time, progress=90, slot=slot))
            conclusion_response = "An error occurred while processing the decision tree. Unable to provide a full conclusion."
        # --- End of decision tree processing ---

        print(OutputFormatter.color_prefix("Evaluating the need for a long response...", "Internal",
                                           time.time() - start_time, progress=94, slot=slot))
        evaluation_prompt = f"""
        {prompt}\n
        Based on: '{user_input}', initial response '{initial_response}', and conclusion '{conclusion_response}',
        does the query require a long response? Respond in JSON format with 'yes' or 'no'.
        ```json
        {{
            "decision": ""
        }}
        ```
        Generate JSON, and only JSON, with the above format.
        """

        evaluation_prompt_tokens = len(TOKENIZER.encode(evaluation_prompt))
        print(OutputFormatter.color_prefix("Processing Evaluation Prompt", "Internal", time.time() - start_time,
                                           token_count=evaluation_prompt_tokens, progress=95, slot=slot))

        evaluation_response = self.invoke_llm(evaluation_prompt)  # Corrected call

        # --- Use process_json_response for evaluation ---
        evaluation_json = self.process_json_response(evaluation_response)
        requires_long_response = False  # Default value
        if evaluation_json:
            requires_long_response = evaluation_json.get("decision", "no").lower() == "yes"
        # --- End of JSON processing ---

        if not requires_long_response:
            print(OutputFormatter.color_prefix("Determined a short response is sufficient...", "Internal",
                                               time.time() - start_time, progress=98, slot=slot))
            asyncio.run_coroutine_threadsafe(
                self.database_manager.async_db_write(slot, user_input, conclusion_response), loop)
            end_time = time.time()
            generation_time = end_time - start_time
            print(OutputFormatter.color_prefix(conclusion_response, "Adelaide", generation_time,
                                               token_count=prompt_tokens, slot=slot))
            # --- Add to context and embed NOW (for short, deep-thought responses) ---
            self.partition_context.add_context(slot, conclusion_response, "main")
            asyncio.run_coroutine_threadsafe(
                self.partition_context.async_embed_and_store(conclusion_response, slot, "main"), loop)
            # ---

            return conclusion_response

        print(OutputFormatter.color_prefix("Handling a long response...", "Internal", time.time() - start_time,
                                           progress=98, slot=slot))
        long_response_estimate_prompt = f"{prompt}\nEstimate tokens needed for a detailed response to '{user_input}'. Respond with JSON, and only JSON, in this format:\n```json\n{{\"tokens\": <number of tokens>}}\n```"

        long_response_estimate_prompt_tokens = len(TOKENIZER.encode(long_response_estimate_prompt))
        print(OutputFormatter.color_prefix("Processing Long Response Estimate Prompt", "Internal",
                                           time.time() - start_time, token_count=long_response_estimate_prompt_tokens,
                                           progress=99, slot=slot))

        long_response_estimate = self.invoke_llm(long_response_estimate_prompt)  # Corrected call

        # --- Use process_json_response for token estimate ---
        tokens_estimate_json = self.process_json_response(long_response_estimate)
        required_tokens = 500  # Default value
        if tokens_estimate_json:
            try:
                required_tokens = int(tokens_estimate_json.get("tokens", 500))
            except (ValueError, TypeError):
                print(OutputFormatter.color_prefix("Failed to parse token estimate. Defaulting to 500 tokens.",
                                                   "Internal"))
        # --- End of JSON processing ---

        print(OutputFormatter.color_prefix(f"Estimated tokens needed: {required_tokens}", "Internal",
                                           time.time() - start_time, progress=99, slot=slot))

        long_response = ""
        remaining_tokens = required_tokens
        continue_prompt = "Continue the response, maintaining coherence and relevance."

        if stream:
            print(
                OutputFormatter.color_prefix("Streaming is enabled, but not yet implemented. Returning full response.",
                                             "Internal"))
            while remaining_tokens > 0:
                part_response_prompt = f"{prompt}\n{continue_prompt}"
                part_response = self.invoke_llm(part_response_prompt)  # Corrected call
                long_response += part_response
                remaining_tokens -= len(TOKENIZER.encode(part_response))
                prompt = f"{prompt}\n{part_response}"
                if remaining_tokens > 0:
                    time.sleep(2)

            asyncio.run_coroutine_threadsafe(self.database_manager.async_db_write(slot, user_input, long_response),
                                             loop)
            end_time = time.time()
            generation_time = end_time - start_time
            # --- Add to context and embed NOW (for streamed responses) ---
            self.partition_context.add_context(slot, long_response, "main")
            asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(long_response, slot, "main"),
                                             loop)
            # ---

            return long_response
        else:
            while remaining_tokens > 0:
                print(OutputFormatter.color_prefix(
                    f"Generating part of the long response. Remaining tokens: {remaining_tokens}...", "Internal",
                    time.time() - start_time, progress=99, slot=slot))
                part_response_prompt = f"{prompt}\n{continue_prompt}"

                part_response_prompt_tokens = len(TOKENIZER.encode(part_response_prompt))
                print(OutputFormatter.color_prefix("Processing Part Response Prompt", "Internal",
                                                   time.time() - start_time, token_count=part_response_prompt_tokens,
                                                   progress=99, slot=slot))

                part_response = self.invoke_llm(part_response_prompt)  # Corrected call
                long_response += part_response

                remaining_tokens -= len(TOKENIZER.encode(part_response))

                prompt = f"{prompt}\n{part_response}"

                if remaining_tokens > 0:
                    time.sleep(2)

            print(OutputFormatter.color_prefix("Completed generation of the long response.", "Internal",
                                               time.time() - start_time, progress=100, slot=slot))
            asyncio.run_coroutine_threadsafe(self.database_manager.async_db_write(slot, user_input, long_response),
                                             loop)
            end_time = time.time()
            generation_time = end_time - start_time
            print(OutputFormatter.color_prefix(long_response, "Adelaide", generation_time, token_count=prompt_tokens,
                                               slot=slot))
            # --- Add to context and embed NOW (for long responses) ---
            self.partition_context.add_context(slot, long_response, "main")
            asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(long_response, slot, "main"),
                                             loop)

            return long_response

    def calculate_total_context_length(self, slot, requester_type):
        """Calculates the total context length for a given slot and requester type."""
        return self.partition_context.calculate_total_context_length(slot, requester_type)


class JSONInterpreter:
    """
    A class to interpret, validate, and self-repair JSON strings.
    """

    def __init__(self, llm_invoker):
        """
        Initializes the JSONInterpreter.

        Args:
            llm_invoker: A function that can invoke the LLM (e.g., ai_runtime_manager.invoke_llm).
        """
        self.llm_invoke = llm_invoker

    def _extract_json(self, text: str) -> str:
        """Extracts a JSON string from text using regex, handling various cases."""
        # Match JSON objects that are complete
        match = re.search(r"\{(?:[^{}]|(?:\".*?\")|(?:\{(?:[^{}]|(?:\".*?\"))*\}))*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        # Handle cases where JSON is incomplete by finding the largest valid JSON object
        start = text.find('{')
        if start == -1: return ""  # No JSON object found

        balance = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == '{':
                balance += 1
            elif text[i] == '}':
                balance -= 1
            if balance == 0 and text[i] == '}':
                end = i + 1
                break

        if end != -1:
            return text[start:end]
        else:
            return text[start:]  # Return partial JSON if no complete object found

    def _find_parsing_errors(self, json_string: str) -> Optional[list]:
        """Identifies specific parsing errors in the JSON string."""
        try:
            json.loads(json_string)
            return None  # No errors
        except json.JSONDecodeError as e:
            errors = []
            errors.append(f"General error: {e.msg} at line {e.lineno}, column {e.colno}")

            # Check for unescaped control characters
            if "Unterminated string" in e.msg:
                errors.append("Possible unterminated string.  Check for missing quotes or escaped characters.")
            if "Expecting value" in e.msg and e.pos == len(json_string.strip()) - 1:
                errors.append("JSON appears incomplete.  Check for missing closing braces or brackets.")
            if "Expecting ',' delimiter" in e.msg:
                errors.append(
                    "Missing comma between elements.  Check that elements in arrays and objects are separated by commas")

            # Check for missing closing brackets/braces
            open_brackets = json_string.count('{') + json_string.count('[')
            close_brackets = json_string.count('}') + json_string.count(']')
            if open_brackets > close_brackets:
                errors.append(
                    f"Missing closing brackets/braces.  Found {open_brackets} opening, but {close_brackets} closing.")

            return errors

    def _basic_json_repair(self, json_string: str) -> str:
        """
        Attempts basic JSON repairs without using an LLM.  Handles:
        1. Incomplete JSON (missing closing braces/brackets).
        2. Trailing commas
        3. Basic missing quotes
        4. Add missing commas.
        """
        json_string = json_string.strip()

        # 1. Handle Incomplete JSON (Missing Braces/Brackets)
        open_brackets = json_string.count('{') + json_string.count('[')
        close_brackets = json_string.count('}') + json_string.count(']')

        if open_brackets > close_brackets:
            missing_brackets = open_brackets - close_brackets
            # Try to intelligently add closing brackets.  This is a heuristic!
            if json_string.count('{') > json_string.count('}'):
                json_string += '}' * missing_brackets
            elif json_string.count('[') > json_string.count(']'):
                json_string += ']' * missing_brackets
            else:  # Just add
                json_string += '}' * missing_brackets

        # 2. Handle Trailing commas
        json_string = re.sub(r',\s*}', '}', json_string)
        json_string = re.sub(r',\s*]', ']', json_string)

        # 3. Basic attempt to add missing quotes around property names (VERY limited)
        json_string = re.sub(r"{\s*([a-zA-Z0-9_]+)\s*:", r'{"\\1":', json_string)
        json_string = re.sub(r",\s*([a-zA-Z0-9_]+)\s*:", r', "\\1":', json_string)

        # 4. Add missing commas
        json_string = re.sub(r'}(\s*)({)', r'}\1,\\2', json_string)
        json_string = re.sub(r'](\s*)(\[)', r']\\1,\\2', json_string)
        json_string = re.sub(r'}(\s*)(")', r'}\1,\\2', json_string)
        json_string = re.sub(r'](\s*)(")', r']\\1,\\2', json_string)
        return json_string

    def repair_json(self, json_string: str, schema: Optional[str] = None, max_retries: int = 3) -> Optional[str]:
        """
        Repairs a JSON string using basic repair methods (no LLM).

        Args:
            json_string: The potentially invalid JSON string.
            schema: Optional JSON schema for validation (not used in this basic repair).
            max_retries: Not used in this version (as there's no LLM retry).

        Returns:
            The repaired JSON string, or None if repair fails.
        """
        json_string = self._extract_json(json_string)  # Clean it
        if not json_string:
            return None

        # --- Attempt basic repair ---
        repaired_json = self._basic_json_repair(json_string)  # Corrected call
        try:
            json.loads(repaired_json)
            print(OutputFormatter.color_prefix("JSON successfully repaired with basic repair.", "Internal"))
            return repaired_json  # Basic repair succeeded!
        except json.JSONDecodeError:
            print(OutputFormatter.color_prefix("Basic JSON repair failed.", "Internal"))
            return None  # Basic repair failed


class PartitionContext:
    def __init__(self, ctx_window_llm, database_manager, vector_store):
        self.ctx_window_llm = ctx_window_llm
        self.db_cursor = database_manager.db_cursor
        self.vector_store = vector_store
        self.L0_size = int(ctx_window_llm * 0.75)  # 75% for L0 (Immediate)
        self.L1_size = int(ctx_window_llm * 0.25)  # 25% for L1 (Semantic)
        self.S_size = 0  # S (Safety Margin) is not used, always 0
        self.context_slots = {}
        """
        Partition Context Management:

        - L0 (Immediate): 75% of the context window. This is the in-memory context that is immediately available to the model.
        - L1 (Semantic): 25% of the context window. This is the context fetched from the database using context-aware embeddings (e.g., Snowflake Arctic).
            - If the context is requested for the 'main' interaction, it is fetched from the 'interaction_history' table.
            - If the context is requested for 'CoT' (Chain of Thought), it is fetched from both the 'interaction_history' and 'CoT_generateResponse_History' tables.
        - S (Safety Margin): 0% of the context window. This is a safety margin and is intentionally left blank. It does not store any context.
        
        Context is managed per slot. Each slot has its own L0, L1, and S partitions.
        """

    def get_context(self, slot, requester_type):
        """
        Retrieves the context for a given slot and requester type.

        Args:
            slot (int): The slot number.
            requester_type (str): The type of requester ('main' or 'CoT').

        Returns:
            list: The context for the specified slot and requester type.
        """
        if slot not in self.context_slots:
            self.context_slots[slot] = {"main": [], "CoT": []}

        if requester_type == "main":
            return self.context_slots[slot]["main"]
        elif requester_type == "CoT":
            return self.context_slots[slot]["main"] + self.context_slots[slot]["CoT"]
        else:
            raise ValueError("Invalid requester type. Must be 'main' or 'CoT'.")

    def add_context(self, slot, text, requester_type):
        """
        Adds context to the specified slot and requester type.

        Args:
            slot (int): The slot number.
            text (str): The context text to add.
            requester_type (str): The type of requester ('main' or 'CoT').
        """
        if slot not in self.context_slots:
            self.context_slots[slot] = {"main": [], "CoT": []}

        context_list = self.context_slots[slot][requester_type]
        context_list.append(text)

        if requester_type == "main":
            self.manage_l0_overflow(slot)

    def manage_l0_overflow(self, slot):
        """
        Manages L0 overflow by truncating or demoting to L1 (database).
        """
        l0_context = self.context_slots[slot]["main"]
        l0_tokens = sum([len(TOKENIZER.encode(item)) for item in l0_context if isinstance(item, str)])

        while l0_tokens > self.L0_size:
            overflowed_item = l0_context.pop(0)
            l0_tokens -= len(TOKENIZER.encode(overflowed_item))
            # Demote to CoT table (it's still in the DB, just not 'main').
            asyncio.run_coroutine_threadsafe(self.async_store_CoT_generateResponse(overflowed_item, slot), loop)

    def get_relevant_chunks(self, query, slot, k=5, requester_type="main"):
        start_time = time.time()
        try:
            # --- Embed the query using embedding_llm ---
            query_embedding = embedding_llm.embed(query)

            # --- Fetch chunks from the database ---
            if requester_type == "CoT":
                table_name = "CoT_generateResponse_History"
            else:
                table_name = "interaction_history"

            self.db_cursor.execute(
                f"SELECT chunk, embedding FROM {table_name} WHERE slot = ?", (slot,)
            )
            rows = self.db_cursor.fetchall()

            # --- Calculate cosine similarities ---
            similarities = []
            chunks = []
            for text_content, pickled_embedding in rows:
                try:
                    embedding = pickle.loads(pickled_embedding)
                    # --- Robustness: Handle potential type errors ---
                    if isinstance(embedding, float):  # Corrected variable name here
                        embedding = [embedding]
                    if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
                        print(f"Warning: Invalid embedding format for chunk: {text_content[:50]}...")
                        continue  # Skip

                    similarity = \
                    cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(embedding).reshape(1, -1))[0][
                        0]
                    similarities.append(similarity)
                    chunks.append(text_content)
                except Exception as e:
                    print(f"Error processing chunk {text_content[:50]}...: {e}")
                    continue

            # --- Get top-k chunks.
            if chunks:  # Ensure we have valid chunks before sorting.
                top_k_indices = np.argsort(similarities)[::-1][:k]
                relevant_chunks = [(chunks[i], similarities[i]) for i in top_k_indices]
            else:
                relevant_chunks = []

            print(OutputFormatter.color_prefix(
                f"Retrieved {len(relevant_chunks)} relevant chunks from database in {time.time() - start_time:.2f}s",
                "Internal"))
            return relevant_chunks

        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error in retrieve_relevant_chunks: {e}", "Internal",
                                               time.time() - start_time))
            traceback.print_exc()
            return []

    # def combine_results(self, list1, list2, k): # REMOVED

    def combine_results(self, list1, list2, k):
        """
        Combines two lists of (doc, score) tuples, removing duplicates and keeping only the top 'k' results based on score.
        """
        combined = {}
        for doc, score in list1 + list2:
            if doc.metadata['doc_id'] not in combined or combined[doc.metadata['doc_id']][1] > score:
                combined[doc.metadata['doc_id']] = (doc, score)
        return sorted(combined.values(), key=lambda x: x[1])[:k]

    async def async_embed_and_store(self, text_chunk, slot, requester_type):
        async with db_lock:
            try:
                if text_chunk is None:
                    print(OutputFormatter.color_prefix("Warning: Received None in async_embed_and_store. Skipping.",
                                                       "Internal"))
                    return

                if not isinstance(text_chunk, str):
                    print(OutputFormatter.color_prefix(
                        f"Warning: Non-string content in async_embed_and_store: {type(text_chunk)}. Skipping.",
                        "Internal"))
                    return

                texts = [text_chunk]  # Embed the whole chunk
                texts = [str(t) for t in texts]  # Ensure they're strings

                for text in texts:
                    # Use the embedding thread.
                    doc_id = str(time.time())  # Unique ID *before* adding to queue
                    embedding_thread.embed_and_store(text, slot, requester_type, doc_id)

            except Exception as e:
                print(OutputFormatter.color_prefix(f"Error in embed_and_store: {e}", "Internal"))
                traceback.print_exc()

    async def async_store_CoT_generateResponse(self, message, slot):
        """Asynchronously stores data in CoT_generateResponse_History."""
        async with db_lock:
            try:
                doc_id = str(time.time())  # Generate a unique doc_id
                db_writer.schedule_write(
                    "INSERT INTO CoT_generateResponse_History (slot, message, doc_id, chunk, embedding) VALUES (?, ?, ?, ?, ?)",
                    (slot, message, doc_id, message, pickle.dumps([])),  # Pass doc_id and chunk
                )
                print(OutputFormatter.color_prefix(
                    f"Stored CoT_generateResponse message for slot {slot}: {message[:50]}...", "Internal"))

            except Exception as e:
                print(OutputFormatter.color_prefix(f"Error storing CoT_generateResponse message: {e}", "Internal"))

    def get_next_task(self):
        with self.lock:
            """Gets the next task from the highest priority queue that is not empty."""
            print(
                OutputFormatter.color_prefix(f"get_next_task: task_queue = {self.task_queue}", "Internal"))  # Debugging
            print(OutputFormatter.color_prefix(f"get_next_task: backbrain_tasks = {self.backbrain_tasks}",
                                               "Internal"))  # Debugging
            if hasattr(self, 'mesh_network_tasks'):
                print(OutputFormatter.color_prefix(f"get_next_task: mesh_network_tasks = {self.mesh_network_tasks}",
                                                   "Internal"))  # Debugging

    def calculate_total_context_length(self, slot, requester_type):
        """Calculates the total context length for a given slot and requester type."""
        context = self.get_context(slot, requester_type)
        total_length = sum([len(TOKENIZER.encode(item)) for item in context if isinstance(item, str)])
        return total_length


class LiteratureReviewer:
    @staticmethod
    def literature_review(query):
        """Simulates performing a literature review."""
        print(OutputFormatter.color_prefix(f"Performing literature review for query: {query}", "Internal"))
        return "This is a placeholder for the literature review results."


class DecisionTreeProcessor:
    @staticmethod
    def process_node(node, prompt, start_time, progress_interval, partition_context, slot):
        """Processes a single node in the decision tree."""
        node_id = node["node_id"]
        node_type = node["node_type"]
        content = node["content"]

        prompt_tokens = len(TOKENIZER.encode(prompt))

        print(OutputFormatter.color_prefix(f"Processing node: {node_id} ({node_type}) - {content}", "Internal",
                                           generation_time=time.time() - start_time, token_count=prompt_tokens,
                                           progress=progress_interval, slot=slot))

        if node_type == "question":
            question_prompt = f"{prompt}\nQuestion: {content}\nAnswer:"  # Constructs a prompt
            question_prompt_tokens = len(TOKENIZER.encode(question_prompt))
            response = ai_runtime_manager.invoke_llm(question_prompt)  # Gets answer
            # Store ALL decision tree node results in CoT context.
            partition_context.add_context(slot, response, "CoT")
            asyncio.run_coroutine_threadsafe(partition_context.async_embed_and_store(response, slot, "CoT"),
                                             loop)  # Embeds result
            print(OutputFormatter.color_prefix(f"Response to question: {response}", "Internal",
                                               generation_time=time.time() - start_time,
                                               token_count=question_prompt_tokens, progress=progress_interval,
                                               slot=slot))
        elif node_type == "action step":  # ADDED
            if "literature_review" in content:
                review_query = re.search(r"literature_review\(['\"](.*?)['\"]\)", content).group(1)
                review_result = LiteratureReviewer.literature_review(review_query)  # Calls
                partition_context.add_context(slot, f"Literature review result for '{review_query}': {review_result}",
                                              "CoT")
                asyncio.run_coroutine_threadsafe(partition_context.async_embed_and_store(
                    f"Literature review result for '{review_query}': {review_result}", slot, "CoT"), loop)
            reflection = "Placeholder for reflection/conclusion"  # Example
            reflection_prompt_tokens = 0
            print(OutputFormatter.color_prefix(f"Reflection/Conclusion: {reflection}", "Internal",
                                               generation_time=time.time() - start_time,
                                               token_count=reflection_prompt_tokens, progress=progress_interval,
                                               slot=slot))

        for option in node.get("options", []):
            print(OutputFormatter.color_prefix(f"Option considered: {option['option_text']}", "Internal",
                                               generation_time=time.time() - start_time, progress=progress_interval,
                                               slot=slot))


def initialize_models():
    """Initializes the LLM and schedules the warmup task."""
    global llm, embedding_llm, ai_runtime_manager, database_manager

    n_batch = 512
    # LLM for text generation (keep existing settings)
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=-1,
        n_batch=n_batch,
        n_ctx=CTX_WINDOW_LLM,
        f16_kv=True,
        verbose=False,
        max_tokens=MAX_TOKENS_GENERATE
    )

    embedding_llm = Llama(model_path=EMBEDDING_MODEL_PATH, n_ctx=CTX_WINDOW_LLM, n_gpu_layers=-1, n_batch=n_batch,
                          embedding=True)

    database_manager = DatabaseManager(DATABASE_FILE, loop)
    ai_runtime_manager = AIRuntimeManager(llm, database_manager)
    database_manager.ai_runtime_manager = ai_runtime_manager
    ai_runtime_manager.llm = llm
    ai_runtime_manager.initialize_gguf_parser()

    # Schedule warmup as a task with priority 0
    print(OutputFormatter.color_prefix("Scheduling LLM warmup...", "Internal"))
    ai_runtime_manager.add_task(warmup_llm, priority=0)  # Use the new warmup_llm function

    return database_manager


def warmup_llm():
    """Performs the LLM warmup."""
    print(OutputFormatter.color_prefix("Warming up the LLM...", "Internal"))
    try:
        warmup_prompt = "Hello"  # SIMPLIFIED PROMPT
        generated_text = ai_runtime_manager.invoke_llm(warmup_prompt, caller="Warmup")
    except Exception as e:
        print(OutputFormatter.color_prefix(f"Error during LLM warmup: {e}", "Internal"))
        traceback.print_exc()
        sys.exit(1)
    print(OutputFormatter.color_prefix(f"Warmup response: {generated_text}", "Internal"))


async def input_task(ai_runtime_manager, partition_context):
    """Task to handle user input in a separate thread (CLI)."""
    current_slot = 0
    while True:
        try:
            user_input = await loop.run_in_executor(None, input, OutputFormatter.color_prefix("", "User"))
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "next slot":
                current_slot += 1
                print(OutputFormatter.color_prefix(f"Switched to slot {current_slot}", "Internal"))
                continue
            # No 'stream' argument for CLI interaction
            ai_runtime_manager.add_task(ai_runtime_manager.generate_response, (user_input, current_slot), 0)

        except EOFError:
            print("EOF")
        except KeyboardInterrupt:
            print(OutputFormatter.color_prefix("\nExiting gracefully...", "Internal"))
            break


async def add_task_async(ai_runtime_manager, task, args, priority):
    """Helper function to add a task to the scheduler from another thread."""
    ai_runtime_manager.add_task((task, args), priority)


async def main():
    global vector_store, db_writer, database_manager, ai_runtime_manager

    database_manager = initialize_models()

    # Debugging: Print table contents
    database_manager.print_table_contents("interaction_history")
    database_manager.print_table_contents("CoT_generateResponse_History")
    database_manager.print_table_contents("vector_learning_context_embedding")
    print(OutputFormatter.color_prefix("Adelaide & Albert Engine initialized. Interaction is ready!", "Internal"))

    partition_context = PartitionContext(CTX_WINDOW_LLM, database_manager, vector_store)
    ai_runtime_manager.partition_context = partition_context

    # Load vector store from the database after starting the writer task
    # vector_store = load_vector_store_from_db(embedding_model, database_manager.db_cursor)
    # partition_context.vector_store = vector_store

    global embedding_thread
    embedding_thread = EmbeddingThread(EMBEDDING_MODEL_PATH, CTX_WINDOW_LLM, N_BATCH,
                                       database_manager)  # Use the global N_BATCH
    embedding_thread.start()

    # Engine runtime watchdog
    watchdog = Watchdog(sys.argv[0], ai_runtime_manager)
    watchdog.start(loop)  # Pass the main event loop to the Watchdog

    # Start the input task
    asyncio.create_task(input_task(ai_runtime_manager, partition_context))
    # Keep main thread running to support the server
    await asyncio.sleep(float('inf'))


class OpenAISpecRequestHandler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers',
                         'Content-Type, Authorization, openai-organization, openai-version, openai-assistant-app-id')  # Add any other required headers
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response_data = {
                "data": [
                    {
                        "id": "Adelaide-and-Albert-Model",  # Use the actual ID of your model
                        "object": "model",
                        "created": 1686935002,  # Replace with a valid timestamp
                        "owned_by": "user",  # Replace with the appropriate owner
                    }
                ],
                "object": "list"
            }
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": {"message": "Not Found", "type": "invalid_request_error", "code": 404}}).encode(
                    'utf-8'))

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                request_data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(
                    {"error": {"message": "Invalid JSON", "type": "invalid_request_error", "code": 400}}).encode(
                    'utf-8'))
                return

            # Extract necessary information, including 'stream'
            messages = request_data.get('messages', [])
            stream = request_data.get('stream', False)  # Default to False if not provided

            # Add the task to the queue, including the 'stream' parameter
            # Use slot 0 for all server requests.  In a real implementation, you'd need
            # a way to map requests to user sessions/slots.
            ai_runtime_manager.add_task(
                (ai_runtime_manager.generate_response, (messages, 0, stream)), 0  # Pass messages directly.
            )

            # Basic response for non-streaming requests
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header('Access-Control-Allow-Origin', '*')  # CORS
            self.end_headers()
            # Placeholder response; actual response handled by generate_response
            self.wfile.write(json.dumps({
                "id": "chatcmpl-placeholder",  # Use a placeholder, actual ID from task
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "Adelaide-and-Albert-Model",  # Replace
                "choices": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }).encode('utf-8'))
            return  # Return from the function, the rest is handled.


        # elif self.path == "/v1/completions":
        # Implement this endpoint as a basic stub or 404.
        #  self.send_response(404)  # Or return a stub response if you prefer
        #  self.end_headers()

        # Add other endpoints with similar 404 or basic responses
        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": {"message": "Not Found", "type": "invalid_request_error", "code": 404}}).encode(
                    'utf-8'))


def get_valid_port(port_str, default_port=8000):
    """
    Validates a port string and returns a valid integer port.
    Returns the default port if the input is invalid.
    """
    try:
        port = int(port_str)
        if 1024 <= port <= 65535:  # Check for valid port range (non-system ports)
            return port
        else:
            print(OutputFormatter.color_prefix(f"Invalid port number: {port_str}. Using default port {default_port}.",
                                               "ServerOpenAISpec"))
            return default_port
    except (ValueError, TypeError):
        print(OutputFormatter.color_prefix(f"Invalid port value: {port_str}. Using default port {default_port}.",
                                           "ServerOpenAISpec"))
        return default_port


def get_valid_host(host_str, default_host='0.0.0.0'):
    """
    Validates a host string (basic check for now).
    Returns the default host if the input is invalid.
    """
    # Basic validation (you might want to add more robust checks, e.g., using regex)
    if host_str and isinstance(host_str, str) and host_str.strip():
        return host_str.strip()
    else:
        print(OutputFormatter.color_prefix(f"Invalid host value: {host_str}. Using default host {default_host}.",
                                           "ServerOpenAISpec"))
        return default_host


def run_server(port=None, host=None):  # Added parameters with None
    global httpd

    # --- Environment Variable Handling (Prioritized) ---
    env_port = os.environ.get('HTTP_PORT')
    env_host = os.environ.get('HTTP_HOST')

    # Use environment variables if available and valid, otherwise use defaults or provided args.
    if env_port:
        port = get_valid_port(env_port)
    elif port is None:  # Check the parameter
        port = get_valid_port(None)  # Use default port

    if env_host:
        host = get_valid_host(env_host)
    elif host is None:  # Check the parameter
        host = get_valid_host(None)  # Use the default.

    server_address = (host, port)
    httpd = HTTPServer(server_address, OpenAISpecRequestHandler)
    print(OutputFormatter.color_prefix(f"Starting OpenAI-compatible server on {host}:{port}...", "ServerOpenAISpec"))
    httpd.serve_forever()


def signal_handler(sig, frame):
    global httpd, ai_runtime_manager, interrupt_count
    interrupt_count += 1
    print(OutputFormatter.color_prefix(f"\nInterrupt signal received ({interrupt_count}/{MAX_INTERRUPTS})...",
                                       "Internal"))

    if interrupt_count >= MAX_INTERRUPTS:
        print(OutputFormatter.color_prefix("Maximum interrupt count reached. Forcefully exiting...", "Internal"))
        # Forceful exit (OS-specific)
        if platform.system() == "Windows":
            # On Windows, we can use TerminateProcess (very aggressive).
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.TerminateProcess(handle, -1)
        else:
            # On POSIX systems, send SIGKILL to the current process.
            os.kill(os.getpid(), signal.SIGKILL)
        sys.exit(1)  # Fallback exit.
    else:
        print(OutputFormatter.color_prefix("Shutting down server...", "Internal"))
        if httpd:
            httpd.shutdown()
        if ai_runtime_manager:
            ai_runtime_manager.unload_model()
        print(OutputFormatter.color_prefix("Server shut down.", "Internal"))
        sys.exit(0)


if __name__ == "__main__":

    system_info = SystemInfoCollector.generate_startup_banner()
    print(OutputFormatter.format_system_info(system_info))

    # --- Clone necessary tools ---
    AIModelPreRuntimeManager.cloning_tools()

    # --- Download and convert models at startup ---
    llm_model_path = AIModelPreRuntimeManager.prepare_llm_model(
        repo_id="MBZUAI/LLaVA-Phi-3-mini-4k-instruct",
        model_name="llava-phi-3",
    )

    # CHANGED EMBEDDING MODEL PATH:
    embedding_model_path = "./Model/ModelCompiledRuntime/snowflake-arctic-embed.gguf"

    sd_model_path = AIModelPreRuntimeManager.prepare_stable_diffusion_model(
        repo_id="stabilityai/stable-diffusion-2-1",
        model_name="stable-diffusion-2-1"
    )
    # --- End of model download/conversion ---

    # Override environment variables
    os.environ["LLM_MODEL_PATH"] = llm_model_path
    if embedding_model_path:
        os.environ["EMBEDDING_MODEL_PATH"] = embedding_model_path
    # --- End of model download/conversion ---

    # Override environment variables
    os.environ["LLM_MODEL_PATH"] = llm_model_path  # Corrected this
    if embedding_model_path:
        os.environ["EMBEDDING_MODEL_PATH"] = embedding_model_path

    signal.signal(signal.SIGINT, signal_handler)

    server_thread = Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print(OutputFormatter.color_prefix("\nExiting gracefully...", "Internal"))
    finally:
        if database_manager:
            database_manager.close()
        loop.close()
        print(OutputFormatter.color_prefix("Cleanup complete. Goodbye!", "Internal"))
