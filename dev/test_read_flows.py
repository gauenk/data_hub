"""

Efficiently read a subset of optical flow

"""
import os
import torch
import numpy as np
from pathlib import Path
from dev_basics.utils.timer import ExpTimer,TimeIt
from easydict import EasyDict as edict
import cache_io

# -- flows --
from dev_basics import flow_io


def run_exp(cfg):

    # -- pick random video --
    vid_names = ["water-slide","koala","tractor-sand","tandem","swing"]
    # vid_names = [p.name for p in Path("flows/480p/").iterdir()]
    index = np.random.permutation(len(vid_names))[0]
    vid_name = vid_names[index]
    # vid_name = cfg.vid_name
    # print(vid_name)

    # -- load --
    timer = ExpTimer()
    flow_path = "flows/480p/%s/" % vid_name
    name = "g-30_seed-123"

    with TimeIt(timer,"flow_io"):
        flows = flow_io.read_flows(flow_path,name,mmap_mode=cfg.mmap_mode,itype="npz")

    with TimeIt(timer,"read_fwd"):
        fflow = flows['fflow'][:cfg.T,:,:cfg.N,:cfg.N].copy()
        # print(np.any(np.isinf(fflow)),fflow.max(),fflow.min(),fflow.shape)
    with TimeIt(timer,"read_bwd"):
        bflow = flows['bflow'][:cfg.T,:,:cfg.N,:cfg.N].copy()
        # print(np.any(np.isinf(bflow)),bflow.max(),bflow.min(),bflow.shape)

    results = edict()
    for key,val in timer.items():
        if "timer_" in key: key = key.split("timer_")[1]
        results[key] = val
    return results

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- configs --
    exps_cfg = {#"group0":{"vid_name":["koala"]},
                "group1":{"N":[128,512]},
                "group2":{"T":[1,30]},
                "group3":{"mmap_mode":["r",None]},
                "cfg":{"seed":0}}
    cache_name = ".cache_io/test_read_flows"
    records_fn = ".cache_io_pkl/test_read_flows.pkl"
    exps = cache_io.exps.unpack(exps_cfg)
    exps,uuids = cache_io.get_uuids(exps,cache_name)
    clear_fxn = lambda x,y: False
    results = cache_io.run_exps(exps,run_exp,uuids=uuids,
                                name=cache_name,enable_dispatch="slurm",
                                records_fn=records_fn,skip_loop=False,
                                records_reload=True,clear=True,
                                clear_fxn=clear_fxn)
    results['total'] = results['read_fwd'] + results['read_bwd'] + results['flow_io']
    print(results[['total','T','N','mmap_mode']])


if __name__ == "__main__":
    main()
