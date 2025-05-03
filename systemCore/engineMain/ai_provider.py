# ai_provider.py
import os
import time
from typing import Any, Dict, Optional
from loguru import logger

# Import necessary Langchain/Provider components
# Ollama
try:
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
    except ImportError:
        logger.error("Failed to import Ollama. Did you install 'langchain-ollama'?")
        ChatOllama = None; OllamaEmbeddings = None
# Fireworks
try:
    from langchain_fireworks import ChatFireworks, FireworksEmbeddings
except ImportError:
    logger.warning("Failed to import Fireworks. Did you install 'langchain-fireworks'?")
    ChatFireworks = None; FireworksEmbeddings = None

# Import necessary config values
try:
    from config import (
        OLLAMA_BASE_URL, OLLAMA_EMBEDDINGS_MODEL, MODEL_ROUTER, MODEL_VLM,
        MODEL_LATEX, MODEL_MATH, MODEL_CODE, MODEL_DEFAULT_CHAT, MODEL_GENERAL_FAST, MODEL_TRANSLATOR,
        MAX_TOKENS, FIREWORKS_API_KEY, FIREWORKS_EMBEDDINGS_MODEL,
        FIREWORKS_CHAT, FIREWORKS_VISUAL_CHAT
        # Add any other config vars needed by AIProvider if they exist
    )
except ImportError as e:
    logger.critical(f"Failed to import config values into ai_provider.py: {e}")
    # Handle inability to get config - maybe exit or set defaults
    raise

# --- PASTE THE AIProvider CLASS DEFINITION HERE ---
# === AI Provider Setup ===
class AIProvider:
    """Handles initialization of multiple LLMs and embedding providers."""
    def __init__(self, provider_name):
        self.provider_name = provider_name
        # Store models in a dictionary
        self.models: Dict[str, Any] = {} # name -> model instance
        self.embeddings = None
        self.EMBEDDINGS_MODEL_NAME = None # Store configured name
        logger.info(f"🤖 Initializing AI Provider: {provider_name}")
        self.setup_provider()

    def setup_provider(self):
        logger.info(f"🔌 Configuring Ollama connection to: {OLLAMA_BASE_URL}")
        start_time = time.time()
        try:
            if self.provider_name == "ollama":
                # --- Ollama Setup ---
                self.EMBEDDINGS_MODEL_NAME = OLLAMA_EMBEDDINGS_MODEL
                # --- Use the imported variable from config.py ---
                logger.info(f"🔌 Configuring Ollama connection to: {OLLAMA_BASE_URL}")
                self.embeddings = OllamaEmbeddings(
                    model=self.EMBEDDINGS_MODEL_NAME,
                    base_url=OLLAMA_BASE_URL # <<< USE THE IMPORTED VARIABLE
                )
                # --- End Modification ---

                model_names = {
                    "router": MODEL_ROUTER, "vlm": MODEL_VLM, "latex": MODEL_LATEX,
                    "math": MODEL_MATH, "code": MODEL_CODE, "general": MODEL_DEFAULT_CHAT,
                    "translator": MODEL_TRANSLATOR,
                    "default": MODEL_DEFAULT_CHAT,
                    "general_fast": MODEL_GENERAL_FAST
                }
                default_temperature = 1.2 # Assuming default temp setting logic here
                logger.info(f"🔥 Setting default LLM temperature to: {default_temperature}")
                common_params = {"temperature": default_temperature}
                if MAX_TOKENS:
                    common_params["max_tokens"] = MAX_TOKENS
                    logger.info(f"🔧 Applying max_tokens limit: {MAX_TOKENS}")

                for key, ollama_name in model_names.items():
                    if not ollama_name:
                        logger.warning(f"Model name for '{key}' not configured in config.py.")
                        continue
                    try:
                        is_vlm = (key == "vlm")
                        if is_vlm:
                             logger.debug(f"  Loading Ollama VLM model '{ollama_name}' for key '{key}'...")
                        else:
                             logger.debug(f"  Loading Ollama model '{ollama_name}' for key '{key}'...")

                        # --- Use the imported variable from config.py ---
                        model_instance = ChatOllama(
                            model=ollama_name,
                            base_url=OLLAMA_BASE_URL, # <<< USE THE IMPORTED VARIABLE
                            **common_params
                        )
                        # --- End Modification ---

                        self.models[key] = model_instance
                        logger.info(f"  ✅ Loaded model '{key}': {ollama_name}")
                    except Exception as model_load_err:
                        logger.error(f"  ❌ Failed to load Ollama model '{ollama_name}' for key '{key}': {model_load_err}")
                        self.models[key] = None

            elif self.provider_name == "fireworks":
                 # --- Fireworks Setup ---
                 self.EMBEDDINGS_MODEL_NAME = FIREWORKS_EMBEDDINGS_MODEL
                 self.embeddings = FireworksEmbeddings(model=self.EMBEDDINGS_MODEL_NAME, fireworks_api_key=FIREWORKS_API_KEY)

                 # --- MODIFICATION: Set default temperature here ---
                 default_temperature = 1.2
                 logger.info(f"🔥 Setting default LLM temperature to: {default_temperature} for Fireworks")
                 fireworks_common_params = {"temperature": default_temperature}
                 if MAX_TOKENS:
                     fireworks_common_params["max_tokens"] = MAX_TOKENS
                     logger.info(f"🔧 Applying max_tokens limit: {MAX_TOKENS} for Fireworks")
                 # --- End Modification ---

                 # Apply the default temp to relevant models
                 # Ensure FIREWORKS_CHAT and FIREWORKS_VISUAL_CHAT are defined in config
                 if FIREWORKS_CHAT:
                    logger.debug(f"  Loading Fireworks model '{FIREWORKS_CHAT}' for key 'default' with temp {default_temperature}...")
                    self.models["default"] = ChatFireworks(
                        model=FIREWORKS_CHAT,
                        fireworks_api_key=FIREWORKS_API_KEY,
                        **fireworks_common_params # Pass temp/max_tokens here
                    )
                    self.models["router"] = self.models["default"] # Router uses default chat model
                    self.models["general"] = self.models["default"]
                 else:
                      logger.error("FIREWORKS_CHAT model name not configured.")
                      self.models["default"] = None; self.models["router"] = None; self.models["general"] = None

                 if FIREWORKS_VISUAL_CHAT:
                     logger.debug(f"  Loading Fireworks VLM model '{FIREWORKS_VISUAL_CHAT}' for key 'vlm' with temp {default_temperature}...")
                     # Note: VLM temp behavior might differ, but setting for consistency
                     self.models["vlm"] = ChatFireworks(
                         model=FIREWORKS_VISUAL_CHAT,
                         fireworks_api_key=FIREWORKS_API_KEY,
                         # Apply temp/max_tokens if supported/desired for VLM
                         **fireworks_common_params
                     )
                 else:
                      logger.warning("FIREWORKS_VISUAL_CHAT model name not configured.")
                      self.models["vlm"] = None

                 # Add other specialist models if defined for Fireworks, applying temp
                 # Example (assuming config vars exist):
                 # self.models["code"] = ChatFireworks(model=FIREWORKS_CODE_MODEL, ..., **fireworks_common_params)
                 # self.models["math"] = ChatFireworks(model=FIREWORKS_MATH_MODEL, ..., **fireworks_common_params)
                 # Set placeholders for now if not defined
                 self.models.setdefault("code", None)
                 self.models.setdefault("math", None)
                 self.models.setdefault("latex", None)
                 self.models.setdefault("translator", None) # Placeholder

            else:
                raise ValueError(f"❌ Invalid provider: {self.provider_name}")

            # --- FIX: Assign self.model and self.vmodel explicitly ---
            self.model = self.models.get("default") # Assign the default model to self.model
            self.vmodel = self.models.get("vlm")    # Assign the vlm model to self.vmodel
            # --- END FIX ---

            # Check if essential models loaded
            if not self.embeddings: raise ValueError("Embeddings failed.")
            if not self.model: raise ValueError("Default chat model (self.model) failed to initialize.") # Check self.model
            if not self.vmodel: logger.warning("⚠️ VLM model (self.vmodel) failed to initialize or is not configured.")

            logger.success(f"✅ AI Provider '{self.provider_name}' setup complete with default temp {default_temperature}.")

        except Exception as e:
            logger.error(f"❌ Error setting up AI provider {self.provider_name}: {e}")
            raise
        finally:
            duration = (time.time() - start_time) * 1000
            logger.debug(f"⏱️ AI Provider setup took {duration:.2f} ms")

    # get_model method remains useful
    def get_model(self, model_key: str = "default") -> Optional[Any]:
        """Gets a specific loaded model instance by its key."""
        model_instance = self.models.get(model_key)
        if not model_instance:
             logger.error(f"Model for key '{model_key}' not available or failed to load.")
             if model_key != "default":
                  logger.warning(f"Falling back to default model.")
                  return self.models.get("default")
        return model_instance

    def get_model(self, model_key: str = "default") -> Optional[Any]:
        """Gets a specific loaded model instance by its key."""
        model_instance = self.models.get(model_key)
        if not model_instance:
             logger.error(f"Model for key '{model_key}' not available or failed to load.")
             # Fallback to default if requested model is missing?
             if model_key != "default":
                  logger.warning(f"Falling back to default model.")
                  return self.models.get("default")
        return model_instance


# --- END PASTE ---