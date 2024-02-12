from typing import List

from beartype import beartype
from braindecode.augmentation import (ChannelsDropout, ChannelsShuffle,
                                      FTSurrogate, GaussianNoise,
                                      SmoothTimeMask)
from torch.nn import Module


@beartype
def create_transforms() -> List[Module]:
    transforms = [
        FTSurrogate(0.5),
        ChannelsDropout(0.5),
        ChannelsShuffle(0.5),
        SmoothTimeMask(0.5),
        ChannelsShuffle(
            probability=0.5,
            p_shuffle=0.1,
        ),
        GaussianNoise(0.5),
    ]

    return transforms
