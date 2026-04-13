from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class SpectralWasteDataset(CustomDataset):
    """SpectralWaste dataset."""

    CLASSES = (
        "background",
        "film",
        "basket",
        "cardboard",
        "video_tape",
        "filament",
        "bag",
    )
    PALETTE = [
        [0, 0, 0],
        [218, 247, 6],
        [51, 221, 255],
        [52, 50, 221],
        [202, 152, 195],
        [0, 128, 0],
        [255, 165, 0],
    ]

    def __init__(
        self,
        img_suffix=".png",
        seg_map_suffix=".png",
        reduce_zero_label=False,
        **kwargs,
    ):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )
