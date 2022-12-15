import random
import hashlib
from typing import List

class MultiHashing():

    def __init__(self, N) -> None:
        
        self.N = N
        self._memomask = {}
        
        for n in range(N):
            self._hash_function(n)
            
        self.lookupDict = {}

    def _hash_function(self, n):
        mask = self._memomask.get(n)
        if mask is None:
            random.seed(n)
            self._memomask[n] = random.getrandbits(32)

    
    def _hash_with_mask(self, x, n):
        return int.from_bytes(hashlib.sha1(x.encode('UTF-8')).digest()[:4], 'little') ^ self._memomask[n]
    
    def compute_hashes(self, x:str) -> List[int]:
        if x not in self.lookupDict:
            self.lookupDict[x] = [self._hash_with_mask(x, n) for n in range(self.N)]
        return self.lookupDict[x]
        
        