
import os
import pickle
import hashlib
import json
import threading
from typing import Any, Optional

class SimpleCache:
    """A thread-safe disk-based cache for storing analysis results."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lock = threading.Lock()

    def _get_path(self, key: str) -> str:
        # Use MD5 hash of the key to avoid filename issues
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.pkl")

    def get(self, key: str) -> Optional[Any]:
        path = self._get_path(key)
        if not os.path.exists(path):
            return None
        
        with self.lock:
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache read error for {key}: {e}")
                return None

    def set(self, key: str, value: Any):
        path = self._get_path(key)
        with self.lock:
            try:
                # Write to a temporary file first for atomicity
                tmp_path = path + ".tmp"
                with open(tmp_path, 'wb') as f:
                    pickle.dump(value, f)
                os.replace(tmp_path, path)
            except Exception as e:
                print(f"Cache write error for {key}: {e}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

def generate_cache_key(prefix: str, **params) -> str:
    """Generate a stable string key from parameters."""
    # Sort keys for stability
    sorted_params = json.dumps(params, sort_keys=True)
    return f"{prefix}:{sorted_params}"
