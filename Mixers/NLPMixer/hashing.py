import random
import hashlib

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
            mask = self._memomask[n] = random.getrandbits(32)            
    
    def hash_with_mask(self, x:str, n:int):
        if x not in self.lookupDict:
            self.lookupDict[x] = int.from_bytes(hashlib.sha1(x.encode('UTF-8')).digest()[:4], 'little') ^ self._memomask[n]
        
        return self.lookupDict[x]
    
    def compute_hashes(self, x:str) -> list[int]:
        return [self.hash_with_mask(x, n) for n in range(self.N)]