
# -- basic imports --
import os
import glob
import subprocess
from pathlib import Path


# -- for all videos --
for path in glob.glob("videos/*"):

    # -- get name and dirs --
    name = path.split("/")[1]
    idir = Path("images") / name
    print(name)
    if idir.exists(): continue
    else: idir.mkdir()

    # -- extract frames --
    fn_mp4 = glob.glob(str(path) + "/*")[0]
    cmd = r"ffmpeg -i "+fn_mp4+r" -framerate 60 "+str(idir)+r"/%04d.png"
    p = subprocess.run(cmd,shell=True,capture_output=True)
    # print(p.stderr)
    # print(p)

    # -- limit to 50 --
    for fn in idir.iterdir():
        num = int(fn.stem)
        if num > 50:
            os.remove(str(fn))
