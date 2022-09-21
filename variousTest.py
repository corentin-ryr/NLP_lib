# import hashlib

# print(hash("bonjour"))
# x = "bonjour"
# print(hash(x))


# print(int.from_bytes(hashlib.sha1("bonjour".encode('UTF-8')).digest()[:4], 'little'))
# x = "bonjour"
# print(int.from_bytes(hashlib.sha1(x.encode('UTF-8')).digest()[:4], 'little'))


# from rich.console import Console

# console = Console()

# def merge_dict(dict_one, dict_two):
#     merged_dict = dict_one | dict_two
#     console.log(merged_dict, log_locals=True)


# merge_dict({'id': 1}, {'name': 'Ashutosh'})


import torch
import math
# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())