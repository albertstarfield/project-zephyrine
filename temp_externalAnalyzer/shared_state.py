# shared_state.py (New File)
import threading
from loguru import logger 

server_is_busy_event = threading.Event()
logger.info("🚦 Shared 'server_is_busy_event' created.")