# core/kraken_nonce.py - Közös nonce kezelő a Kraken API-hoz

import time
import threading

class KrakenNonceManager:
    """
    Singleton nonce manager hogy elkerüljük az 'Invalid nonce' hibákat.
    Minden Kraken API hívás ugyanazt a növekvő nonce-ot használja.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.last_nonce = 0
        return cls._instance
    
    def get_nonce(self):
        """Get next nonce - guaranteed to be increasing"""
        with self._lock:
            # Microseconds timestamp
            nonce = int(time.time() * 1000000)
            
            # Ensure always increasing
            if nonce <= self.last_nonce:
                nonce = self.last_nonce + 1
                
            self.last_nonce = nonce
            return str(nonce)

# Global instance
nonce_manager = KrakenNonceManager()
