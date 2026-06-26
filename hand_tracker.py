# hand_tracker.py
import numpy as np
import cv2
from collections import deque

# =====================================================================
# MODULE CONFIGURATION CONSTANTS
# =====================================================================
MAX_PRIMARY_DISTANCE = 150.0       # Threshold for Hand 1 and Hand 2 matching
BACKGROUND_GRID_THRESHOLD = 40.0   # Tight threshold for background tracking (hand3+)

class GlobalMemoryGrid:
    def __init__(self, max_history=3):
        self.max_history = max_history
        # Structure: { hand_id_int: deque([(cx, cy), ...], maxlen=3) }
        self.registry = {}
        # Keep track of active coasted masks for Hand 1 and Hand 2 when they go missing
        self.coasted_masks = {1: None, 2: None}
        # Track how many consecutive frames Hand 1 or Hand 2 have been coasting
        self.coast_counters = {1: 0, 2: 0}
        self.MAX_COAST_FRAMES = 5

    def update_track_memory(self, hand_id, centroid):
        """Pushes a new verified centroid into the rolling history list for a hand ID."""
        if hand_id not in self.registry:
            self.registry[hand_id] = deque(maxlen=self.max_history)
        self.registry[hand_id].append(centroid)
        
        # If this is a primary hand and it got a real update, reset its coast counter
        if hand_id in [1, 2]:
            self.coast_counters[hand_id] = 0

    def get_last_known_centroid(self, hand_id):
        """Returns the frame_minus_1 (most recent) centroid for a given hand ID."""
        if hand_id in self.registry and len(self.registry[hand_id]) > 0:
            return self.registry[hand_id][-1]
        return None

    def reset_primary_slot(self, hand_id):
        """Completely clears a primary hand slot when its coasting lifespan expires."""
        if hand_id in self.registry:
            del self.registry[hand_id]
        self.coasted_masks[hand_id] = None
        self.coast_counters[hand_id] = 0

    def purge_dead_tracks(self, active_this_frame_ids):
        """Removes deep background tracks that completely vanished from the scene structure."""
        dead_ids = [hid for hid in self.registry if hid > 2 and hid not in active_this_frame_ids]
        for hid in dead_ids:
            del self.registry[hid]


class HandTrackingEngine:
    def __init__(self, max_history=3):
        """Initializes the persistent state machine context for hand tracking."""
        self.memory_grid = GlobalMemoryGrid(max_history=max_history)
        self.active_tracks = {1: None, 2: None}

    def _get_lowest_leftmost_pixel(self, mask):
        """Finds the leftmost pixel on the absolute lowest row containing mask data."""
        y_indices, x_indices = np.where(mask == 1)
        if len(y_indices) == 0:
            return None
        max_y = np.max(y_indices)
        row_x_indices = x_indices[y_indices == max_y]
        leftmost_x = np.min(row_x_indices)
        return int(leftmost_x), int(max_y)

    def track_hand_identities(self, raw_hand_masks, frame_width, frame_height):
        """
        Stateful tracking algorithm implementing a 3-frame spatial memory grid lookup.
        Includes upfront 80% mutual overlap filter for incoming raw masks, and a 
        post-assignment cross-check to invalidate coasting masks that clash with active hands.
        """
        current_frame_hands = {1: None, 2: None}
        all_detected_blobs = []
        
        BOTTOM_BUFFER_ROWS = 5
        gate_threshold_row = frame_height - BOTTOM_BUFFER_ROWS
        frame_midpoint_x = frame_width // 2

        # 1. Parse and extract all individual mask blobs found by the Segmentation Model
        for raw_mask in raw_hand_masks:
            mask_2d = np.squeeze(raw_mask).astype(np.uint8)
            if mask_2d.ndim > 2:
                mask_2d = mask_2d[:, :, 0]
            if np.sum(mask_2d) == 0:
                continue

            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask_2d)
            for label_idx in range(1, num_labels):
                if stats[label_idx, cv2.CC_STAT_AREA] < 300:
                    continue
                    
                single_blob_mask = (labels_im == label_idx).astype(np.uint8)
                anchor = self._get_lowest_leftmost_pixel(single_blob_mask)
                
                y_indices, x_indices = np.where(single_blob_mask == 1)
                centroid = (int(np.mean(x_indices)), int(np.mean(y_indices)))
                total_pixels = int(stats[label_idx, cv2.CC_STAT_AREA])

                is_eligible_primary = False
                if anchor is not None:
                    _, py = anchor
                    if py >= gate_threshold_row:
                        is_eligible_primary = True

                all_detected_blobs.append({
                    "mask": single_blob_mask,
                    "centroid": centroid,
                    "anchor_x": anchor[0] if anchor else centroid[0],
                    "is_eligible_primary": is_eligible_primary,
                    "total_pixels": total_pixels,
                    "forced_slot": None  
                })

        # STAGE 1: UPFRONT 80% MUTUAL OVERLAP FILTER (Raw Blobs)
        blobs_to_discard = set()
        num_blobs = len(all_detected_blobs)

        for i in range(num_blobs):
            if i in blobs_to_discard:
                continue
            for j in range(i + 1, num_blobs):
                if j in blobs_to_discard:
                    continue
                
                mask_i = all_detected_blobs[i]["mask"]
                mask_j = all_detected_blobs[j]["mask"]
                
                intersection = np.sum((mask_i == 1) & (mask_j == 1))
                if intersection == 0:
                    continue
                    
                overlap_ratio_i = intersection / all_detected_blobs[i]["total_pixels"]
                overlap_ratio_j = intersection / all_detected_blobs[j]["total_pixels"]
                
                if overlap_ratio_i >= 0.80 or overlap_ratio_j >= 0.80:
                    if all_detected_blobs[i]["total_pixels"] >= all_detected_blobs[j]["total_pixels"]:
                        survivor_idx, discard_idx = i, j
                    else:
                        survivor_idx, discard_idx = j, i
                        
                    blobs_to_discard.add(discard_idx)
                    
                    survivor_blob = all_detected_blobs[survivor_idx]
                    survivor_cx = survivor_blob["centroid"][0]
                    
                    # FIX: Do NOT force "is_eligible_primary = True". 
                    # Let it keep the real geometric value calculated during the initial parsing.
                    if survivor_blob["is_eligible_primary"]:
                        if survivor_cx < frame_midpoint_x:
                            survivor_blob["forced_slot"] = 1
                        else:
                            survivor_blob["forced_slot"] = 2

        all_detected_blobs = [b for idx, b in enumerate(all_detected_blobs) if idx not in blobs_to_discard]

        # BOOTSTRAPPING PHASE (If tracking history is completely empty)
        if not self.memory_grid.registry or all(self.memory_grid.get_last_known_centroid(i) is None for i in [1, 2]):
            eligible_primaries = [b for b in all_detected_blobs if b["is_eligible_primary"]]
            if len(eligible_primaries) > 0:
                forced_h1 = [b for b in eligible_primaries if b["forced_slot"] == 1]
                forced_h2 = [b for b in eligible_primaries if b["forced_slot"] == 2]
                
                if forced_h1:
                    hand1_cand = max(forced_h1, key=lambda b: b["total_pixels"])
                else:
                    hand1_cand = min(eligible_primaries, key=lambda b: b["anchor_x"])
                    
                current_frame_hands[1] = hand1_cand["mask"]
                self.active_tracks[1] = hand1_cand["centroid"]
                self.memory_grid.update_track_memory(1, hand1_cand["centroid"])

                remaining = [b for b in eligible_primaries if b is not hand1_cand]
                if remaining:
                    if forced_h2:
                        hand2_cand = max(forced_h2, key=lambda b: b["total_pixels"])
                    else:
                        hand2_cand = max(remaining, key=lambda b: b["anchor_x"])
                        
                    current_frame_hands[2] = hand2_cand["mask"]
                    self.active_tracks[2] = hand2_cand["centroid"]
                    self.memory_grid.update_track_memory(2, hand2_cand["centroid"])
            
            self.memory_grid.coasted_masks[1] = current_frame_hands[1]
            self.memory_grid.coasted_masks[2] = current_frame_hands[2]
            return current_frame_hands, self.active_tracks, {}

        # TRACKING PHASE VIA SEQUENTIAL DISTANCE MEMORY LOOKUPS
        assigned_blob_indices = set()
        current_frame_background_assignments = {}

        # STEP 1: Process and lock the primary tracking slots (Hand 1 and Hand 2)
        for primary_id in [1, 2]:
            last_centroid = self.memory_grid.get_last_known_centroid(primary_id)
            
            if last_centroid is None:
                self.active_tracks[primary_id] = None
                current_frame_hands[primary_id] = None
                continue

            best_dist = float('inf')
            best_blob_idx = -1

            for idx, blob in enumerate(all_detected_blobs):
                if idx in assigned_blob_indices or not blob["is_eligible_primary"]:
                    continue
                if blob["forced_slot"] is not None and blob["forced_slot"] != primary_id:
                    continue
                    
                dist = np.linalg.norm(np.array(last_centroid) - np.array(blob["centroid"]))
                if dist < best_dist and dist < MAX_PRIMARY_DISTANCE:
                    best_dist = dist
                    best_blob_idx = idx

            if best_blob_idx != -1:
                current_frame_hands[primary_id] = all_detected_blobs[best_blob_idx]["mask"]
                self.active_tracks[primary_id] = all_detected_blobs[best_blob_idx]["centroid"]
                self.memory_grid.update_track_memory(primary_id, all_detected_blobs[best_blob_idx]["centroid"])
                self.memory_grid.coasted_masks[primary_id] = all_detected_blobs[best_blob_idx]["mask"]
                assigned_blob_indices.add(best_blob_idx)
            else:
                self.memory_grid.coast_counters[primary_id] += 1
                
                if self.memory_grid.coast_counters[primary_id] <= self.memory_grid.MAX_COAST_FRAMES:
                    potential_coast_mask = self.memory_grid.coasted_masks[primary_id]
                    
                    if potential_coast_mask is not None:
                        # STAGE 2: CROSS-CHECK COASTED MASK AGAINST OTHER ACTIVE PRIMARY HANDS
                        other_primary_id = 2 if primary_id == 1 else 1
                        other_mask = current_frame_hands[other_primary_id]
                        
                        is_clashing_with_active_hand = False
                        if other_mask is not None:
                            intersection = np.sum((potential_coast_mask == 1) & (other_mask == 1))
                            if intersection > 0:
                                coast_pixels = np.sum(potential_coast_mask == 1)
                                other_pixels = np.sum(other_mask == 1)
                                
                                ratio_coast = intersection / coast_pixels if coast_pixels > 0 else 0
                                ratio_other = intersection / other_pixels if other_pixels > 0 else 0
                                
                                if ratio_coast >= 0.80 or ratio_other >= 0.80:
                                    is_clashing_with_active_hand = True

                        if not is_clashing_with_active_hand:
                            self.active_tracks[primary_id] = None
                            current_frame_hands[primary_id] = potential_coast_mask
                        else:
                            self.active_tracks[primary_id] = None
                            current_frame_hands[primary_id] = None
                            self.memory_grid.reset_primary_slot(primary_id)
                    else:
                        self.active_tracks[primary_id] = None
                        current_frame_hands[primary_id] = None
                else:
                    self.active_tracks[primary_id] = None
                    current_frame_hands[primary_id] = None
                    self.memory_grid.reset_primary_slot(primary_id)

        # STEP 2: Map remaining background hands (hand3, hand4, etc.)
        unassigned_blob_indices = [i for i in range(len(all_detected_blobs)) if i not in assigned_blob_indices]
        saved_background_ids = [hid for hid in self.memory_grid.registry.keys() if hid > 2]
        matched_background_ids = set()

        for blob_idx in unassigned_blob_indices:
            if all_detected_blobs[blob_idx]["forced_slot"] is not None:
                continue
                
            blob_centroid = all_detected_blobs[blob_idx]["centroid"]
            best_bg_dist = float('inf')
            best_bg_id = -1

            for bg_id in saved_background_ids:
                if bg_id in matched_background_ids:
                    continue
                last_bg_centroid = self.memory_grid.get_last_known_centroid(bg_id)
                if last_bg_centroid is not None:
                    dist = np.linalg.norm(np.array(blob_centroid) - np.array(last_bg_centroid))
                    if dist < best_bg_dist and dist < BACKGROUND_GRID_THRESHOLD:
                        best_bg_dist = dist
                        best_bg_id = bg_id

            if best_bg_id != -1:
                current_frame_background_assignments[blob_idx] = best_bg_id
                self.memory_grid.update_track_memory(best_bg_id, blob_centroid)
                matched_background_ids.add(best_bg_id)
                assigned_blob_indices.add(blob_idx)

        # STEP 3: Assign entirely new slots / handle empty primary re-entry tracking
        final_unassigned_blobs = [i for i in range(len(all_detected_blobs)) if i not in assigned_blob_indices]
        for blob_idx in final_unassigned_blobs:
            blob = all_detected_blobs[blob_idx]
            
            if blob["is_eligible_primary"] and (self.memory_grid.get_last_known_centroid(1) is None) and (blob["forced_slot"] in [None, 1]):
                current_frame_hands[1] = blob["mask"]
                self.active_tracks[1] = blob["centroid"]
                self.memory_grid.update_track_memory(1, blob["centroid"])
                self.memory_grid.coasted_masks[1] = blob["mask"]
                assigned_blob_indices.add(blob_idx)
                
            elif blob["is_eligible_primary"] and (self.memory_grid.get_last_known_centroid(2) is None) and (blob["forced_slot"] in [None, 2]):
                current_frame_hands[2] = blob["mask"]
                self.active_tracks[2] = blob["centroid"]
                self.memory_grid.update_track_memory(2, blob["centroid"])
                self.memory_grid.coasted_masks[2] = blob["mask"]
                assigned_blob_indices.add(blob_idx)
                
            else:
                if blob["forced_slot"] is not None:
                    continue  
                    
                new_bg_id = 3
                while new_bg_id in self.memory_grid.registry or new_bg_id in current_frame_background_assignments.values():
                    new_bg_id += 1
                    
                current_frame_background_assignments[blob_idx] = new_bg_id
                self.memory_grid.update_track_memory(new_bg_id, blob["centroid"])
                assigned_blob_indices.add(blob_idx)

        self.memory_grid.purge_dead_tracks(set(current_frame_background_assignments.values()))

        background_payload = {
            current_frame_background_assignments[idx]: all_detected_blobs[idx]["mask"] 
            for idx in current_frame_background_assignments
        }

        return current_frame_hands, self.active_tracks, background_payload