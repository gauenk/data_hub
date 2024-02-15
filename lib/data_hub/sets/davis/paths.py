
# import socket
import os
from pathlib import Path
pcname = os.uname().nodename
if "anvil" in pcname:
    base = Path("/home/x-kgauen/Documents/")
elif "USING_DOCKER" in os.environ:
    base = Path("/working/")
else:
    base = Path("/home/gauenk/Documents/")
BASE = base / Path("data/davis/DAVIS/")
IMAGE_PATH = BASE / Path("./JPEGImages/480p/")
IMAGE_SETS = BASE / Path("./ImageSets/2017/")
CROPPED_BASE = BASE / "./cropped"
FLOW_BASE =  BASE / Path("./flows/480p/")

