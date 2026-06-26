import cv2
import numpy as np
from ultralytics import YOLO

class SegmentationEngine:
    def __init__(self, model_name="yolov9e-seg.pt", target_classes=None):
        """
        Initializes a generic vision engine.
        target_classes: dict mapping COCO integer IDs to string names.
                        Example: {41: "cup", 67: "cell phone"}
        """
        print(f"Initializing Retina-Mask Vision Engine with {model_name}...")
        self.model = YOLO(model_name)
        
        # Accept target classes dynamically from the outside application
        self.target_classes = target_classes or {}
        
        # Ensure '0' (person/hand) is always monitored automatically
        self.filter_ids = [0] + list(self.target_classes.keys())

    def _calculate_stable_center(self, binary_mask):
        """
        Calculates the internal 'center' using a distance transform.
        Returns the (x, y) coordinates of the 'deepest' pixel inside the mask.
        """
        # Calculate distance from every foreground pixel to the nearest background pixel
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        # Find the coordinates of the maximum distance value (the pole of inaccessibility)
        _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
        
        # max_loc returns (x, y) coordinates mapping directly to (column, row)
        return max_loc

    def extract_masks(self, frame, width, height):
        """
        Runs high-resolution retina-mask segmentation. 
        Returns hand masks as raw GRIDs, and object masks with full descriptive tuples.
        """
        # Resize input frame to match the requested processing canvas size
        frame_resized = cv2.resize(frame, (width, height))
        
        # --- MODIFICA QUI: Aggiunto device='cpu' per evitare il crash della GT 1030 ---
        results = self.model(
            frame_resized, 
            retina_masks=True, 
            verbose=False, 
            classes=self.filter_ids,
            device='cpu'
        )[0].cpu()
        
        hand_masks = []
        object_masks = []
        
        if results.masks is not None:
            masks = results.masks.data.numpy()
            clss = results.boxes.cls.numpy()
            
            for mask, cls_id in zip(masks, clss):
                cls_id = int(cls_id)
                if cls_id not in self.filter_ids: 
                    continue
                
                # Convert the float retina mask into a strict binary matrix (0 or 1)
                binary_mask = (mask > 0.5).astype(np.uint8)
                
                # --- CONDITIONAL HAND/OBJECT ROUTING PATHS ---
                if cls_id == 0:
                    # HANDS ONLY: Bypasses the center calculation.
                    # Appends only the clean binary matrix array.
                    hand_masks.append(binary_mask)
                else:
                    # TARGET OBJECTS: Calculates the robust spatial depth center position.
                    center_position = self._calculate_stable_center(binary_mask)
                    class_name = self.target_classes[cls_id]
                    
                    # Layout format: [name_object, GRID, center_position]
                    object_masks.append([class_name, binary_mask, center_position])
                    
        return hand_masks, object_masks