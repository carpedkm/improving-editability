import os
import sys
from typing import Dict, List, Any
import numpy as np
from torch.utils.data import Dataset

# Ensure current directory is in sys.path
if './' not in sys.path:
    sys.path.append('./')

class multiedit_DATASET(Dataset):
    """Custom Dataset for handling multi-edit data from JSON input."""
    
    def __init__(self, loaded_json: Dict[str, Any], in_pipeline: bool = False, max_layers: int = 2):
        """
        Initialize the dataset with JSON data.

        Args:
            loaded_json (Dict[str, Any]): Loaded JSON data containing dataset information.
            in_pipeline (bool): Flag to indicate if used in a pipeline (affects bbox scaling).
            max_layers (int): Maximum number of layers to filter the dataset.
        """
        self.in_pipeline = in_pipeline
        self.max_layers = max_layers
        
        # Convert keys to string and filter by max_layers
        self.data = {str(counter): value for counter, (key, value) in enumerate(loaded_json.items()) 
                     if len(value) == max_layers}
        
        # Create index mapper for string-based indexing
        self.idx_mapper = {str(counter): key for counter, (key, _) in enumerate(loaded_json.items()) 
                          if len(loaded_json[key]) == max_layers}
        
        self.resolution = 1024  # Image resolution for bounding box scaling

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a dataset item by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, Any]: Dictionary containing processed dataset item with layers, bboxes, prompts, and classes.
        """
        # Map index to original JSON key
        idx_str = str(idx)
        if idx_str not in self.idx_mapper:
            raise IndexError(f"Index {idx} not found in dataset.")
        
        json_key = self.idx_mapper[idx_str]
        item_data = self.data[idx_str]
        total_layers = len(item_data)

        # Process bounding boxes
        bboxes_float = [layer['coordinates'] for layer in item_data]
        bboxes_pixel = [[int(coord * self.resolution) for coord in bbox] for bbox in bboxes_float]
        
        # Create binary masks for bounding boxes
        bboxes = [np.zeros((self.resolution, self.resolution), dtype=np.uint8) for _ in range(total_layers)]
        for i, bbox in enumerate(bboxes_pixel):
            x_min, y_min, x_max, y_max = bbox
            # Ensure coordinates are within bounds
            x_min, x_max = max(0, x_min), min(self.resolution, x_max)
            y_min, y_max = max(0, y_min), min(self.resolution, y_max)
            # Scale to 255 for pipeline mode, 1 otherwise
            bboxes[i][x_min:x_max, y_min:y_max] = 255 if self.in_pipeline else 1

        # Stack bounding boxes into a single array
        bbox_cat = np.stack(bboxes, axis=0)

        # Extract metadata
        classes = [layer['class'] for layer in item_data]
        background_prompt = item_data[0]['background_prompt']
        local_prompts = [layer['local_prompt'] for layer in item_data]
        global_prompt = item_data[0]['global_prompt']
        background = item_data[0]['background']
        considerable_classes = [", ".join(layer['considerable_classes']) for layer in item_data]

        return {
            'total_layers': total_layers,
            'bboxes': bbox_cat,
            'bboxes_pixel': bboxes_pixel,
            'classes': classes,
            'background_prompt': background_prompt,
            'local_prompts': local_prompts,
            'global_prompt': global_prompt,
            'background': background,
            'considerable_classes': considerable_classes
        }