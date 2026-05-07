from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List


class BaseLoader(ABC):
    
    def __init__(self):
        pass
    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> List:
    
        pass
