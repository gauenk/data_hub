
# import socket
import os
from pathlib import Path
pcname = os.uname().nodename
if "anvil" in pcname:
    base = Path("/home/x-kgauen/")
else:
    base = Path("/home/gauenk/")
BASE = base / Path("Documents/data/bsd500/")
IMAGE_PATH = BASE / Path("./images")
IMAGE_SETS = BASE / Path("./ImageSets/2017/")
CROPPED_BASE = BASE / "./cropped"
FLOW_BASE =  BASE / Path("./flows/480p/")

