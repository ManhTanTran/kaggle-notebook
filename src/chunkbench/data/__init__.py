"""Dataset adapters."""

from chunkbench.data.base import DatasetAdapter
from chunkbench.data.hotpotqa_fullwiki import HotpotQAFullWikiAdapter
from chunkbench.data.qasper import QasperAdapter
from chunkbench.data.synthetic import SyntheticDatasetAdapter
from chunkbench.data.uit_viquad import UITViQuADAdapter
from chunkbench.data.vimqa import ViMQAAdapter

__all__ = [
    "DatasetAdapter",
    "HotpotQAFullWikiAdapter",
    "QasperAdapter",
    "SyntheticDatasetAdapter",
    "UITViQuADAdapter",
    "ViMQAAdapter",
]
