
# import socket
import os
from pathlib import Path
pcname = os.uname().nodename
if "anvil" in pcname:
    base = Path("/home/x-kgauen/")
else:
    base = Path("/home/gauenk/")
BASE = base / Path("Documents/data/Vimeo90K/")
IMAGE_PATH = BASE / Path("./vimeo_septuplet/")
LR4_PATH = BASE / Path("./vimeo_septuplet_matlabLRx4/")
FLOW_BASE =  BASE / Path("./flows/")
IMAGE_SETS = BASE / Path("./image_sets/")

