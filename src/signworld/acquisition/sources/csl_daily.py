"""CSL-Daily (Zhou et al., 2021): daily-life Chinese Sign Language, released after an agreement."""

from .base import Access, ManualDatasetSource, NoSettings


class CSLDailySource(ManualDatasetSource[NoSettings]):
    name = "csl_daily"
    homepage = "https://ustc-slr.github.io/datasets/2021_csl_daily/"
    terms = "CSL Dataset Release Agreement, signed by a full-time staff member (not a student)."
    access = Access.MANUAL
    settings_model = NoSettings
    instructions = (
        "send the signed CSL Dataset Release Agreement to ustc_vslrg@126.com with "
        "zhwg@ustc.edu.cn in copy (see the homepage), then place the files you receive "
        "under <data_root>/csl_daily/raw."
    )
