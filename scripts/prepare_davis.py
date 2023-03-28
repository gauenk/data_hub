
import data_hub
from data_hub.prepare import davis
from pathlib import Path

def main():

    # -- get root --
    davis_paths = data_hub.sets.davis.paths
    base = davis_paths.BASE

    # -- verify directory --
    Path(base).exists()

    # -- cropped sets for training --
    davis.generate_cropped.run(base)

    # -- pre-compute optical flow --
    # davis.precompute_flow.run(base)


if __name__ == "__main__":
    main()
