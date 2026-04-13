# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class ZeroWasteDataset(CustomDataset):
    """ZeroWaste dataset.
    """
    CLASSES = ('background', 'rigid_plastic', 'cardboard', 'metal', 'soft_plastic')
    PALETTE = [[10, 10, 10], [230, 5, 5], [4, 200, 3], [204, 5, 255], [5, 128, 148]]

    def __init__(self,
                 img_suffix='.PNG',    # Changed to your specific suffix (or .png / .jpg)
                 seg_map_suffix='.PNG', # Changed to your specific suffix
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
