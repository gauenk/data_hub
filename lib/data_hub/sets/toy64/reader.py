
def read_video(path,nframes,ext="png"):
    vid = []
    for t in range(nframes):
        path_t = path / ("%05d.%s" % (t,ext))
        if not path_t.exists(): break
        vid_t = Image.open(str(path_t)).convert("RGB")
        vid_t = np.array(vid_t)*1.
        vid_t = rearrange(vid_t,'h w c -> c h w')
        vid.append(vid_t)
    vid = np.stack(vid)
    return vid

def read_files(iroot,sroot,ds_split):

    # -- get vid names in set --
    split_fn = sroot / ("%s.txt" % ds_split)
    vid_names = get_vid_names(split_fn)

    # -- get files --
    files = {'images':{}}
    for vid_name in vid_names:
        vid_dir = iroot/vid_name
        vid_frames = list(vid_dir.iterdir())
        files['images'][vid_name] = vid_frames
    print("-"*30)
    print(files)

    return files
