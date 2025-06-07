# shared_state.py (New File)
import threading
from loguru import logger 

server_is_busy_event = threading.Event()
logger.info("🚦 Shared 'server_is_busy_event' created.")

class TaskInterruptedException(Exception):
    """Custom exception raised when a lower-priority task is interrupted by a higher-priority one."""
    pass
