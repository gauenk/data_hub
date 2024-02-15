import os
from pathlib import Path
pcname = os.uname().nodename
if "anvil" in pcname:
    base = Path("/home/x-kgauen/Documents/")
elif "USING_DOCKER" in os.environ:
    base = Path("/working/")
else:
    base = Path("/home/gauenk/Documents/")
BASE = base / Path("data/set8/")
IMAGE_PATH = BASE / Path("./images/")
IMAGE_SETS = BASE / Path("./image_sets/")
FLOW_BASE =  BASE / Path("./flows/")
