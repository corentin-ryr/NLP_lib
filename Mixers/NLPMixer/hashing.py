import random
from typing import Callable
import hashlib

class MultiHashing():

    def __init__(self, N) -> None:
        
        self._memomask = {}
        self.hashes:list[Callable[[str], list[int]]] = []
        
        for n in range(N):
            self.hashes.append(self._hash_function(n))
            
        self.lookupDict = {}

    def _hash_function(self, n):
        mask = self._memomask.get(n)
        if mask is None:
            random.seed(n)
            mask = self._memomask[n] = random.getrandbits(32)
            
        def myhash(x):
            return int.from_bytes(hashlib.sha1(x.encode('UTF-8')).digest()[:4], 'little') ^ mask

        return myhash
    
    def compute_hashes(self, x:str) -> list[int]:
        if x not in self.lookupDict:
            self.lookupDict[x] = [hashFunc(x) for hashFunc in self.hashes]
        return self.lookupDict[x]
        