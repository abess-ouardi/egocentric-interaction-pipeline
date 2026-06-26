import os
import sys
import cv2
import numpy as np
import pandas as pd

#PUT IN HERE THE PARENT DIRECTORY OF THIS SCRIPT
parent_dir = ""


from vision import SegmentationEngine
from hand_tracker import HandTrackingEngine  

# =====================================================================
# GLOBAL CONFIGURATION & PARAMETERS
# =====================================================================
VIDEO_PATH = os.path.join(parent_dir, "push_cup.mp4")
DEPTH_VIDEO_PATH = os.path.join(parent_dir, "depth_push_cup.mp4")
CSV_OUTPUT_PATH = os.path.join(parent_dir, "validation_checks", "debug_interaction_hybrid_push_cupx1.csv")

# --- SNIPPET 1: ADD VISUALIZATION OUTPUT PATH CONSTANT ---
OUTPUT_VIDEO_PATH = os.path.join(parent_dir, "validation_checks", "hybrid_interaction_matrix_push_cupx1.mp4")

MIN_OVERLAP_PIXELS = 5
DEPTH_DIFF_THRESHOLD = 25.0

TARGET_CLASSES = {
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    63: "laptop", 64: "mouse", 67: "cell phone", 73: "book", 77: "teddy bear"
}

ORDERED_COLUMNS = [
    "Frame",
    "object_hand1",
    "num_pixels_overlapping_hand1",
    "average_depth_overlapping_pixels_hand1",
    "depth_center_object_hand1",
    "absolute_depth_diff_hand1",  # Sits right before object_hand2
    "object_hand2",
    "num_pixels_overlapping_hand2",
    "average_depth_overlapping_pixels_hand2",
    "depth_center_object_hand2",
    "absolute_depth_diff_hand2"   # Appends directly as the final entry of the line
]


# =====================================================================
# CORE IMPLEMENTATION FUNCTIONS
# =====================================================================

def initialize_environment(video_path, depth_video_path):
    """Opens video captures and extracts dimensional sync metrics."""
    cap = cv2.VideoCapture(video_path)
    d_cap = cv2.VideoCapture(depth_video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, d_cap, width, height, total_frames


def load_segmentation_engine(model_name, target_classes):
    """Instantiates the deep learning inference handler."""
    return SegmentationEngine(model_name=model_name, target_classes=target_classes)


def dilate_hand_mask(h_mask, kernel_size=(3, 3)):
    """
    Applies morphological dilation to an individual hand mask array.
    A 3x3 kernel size yields an exact 1-pixel expansion on all sides.
    """
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(h_mask.astype(np.uint8), kernel, iterations=1)


def evaluate_hand_object_interactions(dilated_h_mask, object_masks, depth_gray, width, height):
    """
    Evaluates a dilated hand mask against all active object targets.
    Filters through 2D spatial intersection and 3D physical depth criteria.
    """
    frame_hand_objects = []
    frame_hand_overlaps = []
    frame_hand_avg_depths = []
    frame_hand_center_depths = []
    frame_hand_diffs = []

    for item in object_masks:
        if not item or len(item) == 0: 
            continue
        obj_name, o_mask = (item[0], item[1]) if isinstance(item[0], str) else (item[1], item[0])
        if np.sum(o_mask) == 0: 
            continue

        # Gate 1: 2D Overlap Constraint
        intersection_mask = cv2.bitwise_and(dilated_h_mask, o_mask.astype(np.uint8))
        overlap_pixel_count = int(np.sum(intersection_mask))

        if overlap_pixel_count < MIN_OVERLAP_PIXELS:
            continue

        # Gate 2: 3D Proximity Depth Constraint
        overlap_depth_pixels = depth_gray[intersection_mask == 1]
        overlap_avg_depth = float(np.mean(overlap_depth_pixels))

        contours, _ = cv2.findContours(o_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx, cy = int(width / 2), int(height / 2)
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
        
        cx = min(max(0, cx), width - 1)
        cy = min(max(0, cy), height - 1)
        center_point_depth = float(depth_gray[cy, cx])

        # --- Math logic folded in-line here instead of using external function call ---
        abs_diff_value = abs(overlap_avg_depth - center_point_depth)
        
        if abs_diff_value < DEPTH_DIFF_THRESHOLD:
            display_name = "cellphone" if obj_name == "cell phone" else obj_name
            
            frame_hand_objects.append(display_name)
            frame_hand_overlaps.append(str(overlap_pixel_count))
            frame_hand_avg_depths.append(str(round(overlap_avg_depth, 2)))
            frame_hand_center_depths.append(str(round(center_point_depth, 2)))
            frame_hand_diffs.append(str(round(abs_diff_value, 2)))

    return frame_hand_objects, frame_hand_overlaps, frame_hand_avg_depths, frame_hand_center_depths, frame_hand_diffs


def save_telemetry_report(csv_records, ordered_columns, output_path):
    """Converts records, forces strict layout column ordering, and saves as semi-colon CSV."""
    df_out = pd.DataFrame(csv_records)

    # Reindex layout sequence safely keeping only columns that exist inside dataset context
    df_out = df_out[[col for col in ordered_columns if col in df_out.columns]]

    # Create folders if necessary before initializing text serialization pipelines
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_out.to_csv(output_path, sep=";", index=False)
    print(f"\n[SUCCESS] Custom Matrix with Identity-Preserved updates compiled at: {output_path}\n")


# =====================================================================
# RUNTIME ORCHESTRATION PIPELINE
# =====================================================================

def main():
    print("=" * 95)
    print("LAUNCHING MODULAR IDENTITY-PRESERVED MULTI-OBJECT INTERACTION GENERATOR")
    print("=" * 95)

    # Initialize assets and parsing targets
    cap, d_cap, width, height, total_frames = initialize_environment(VIDEO_PATH, DEPTH_VIDEO_PATH)
    vision_eng = load_segmentation_engine(model_name="yolov9e-seg.pt", target_classes=TARGET_CLASSES)
    tracker = HandTrackingEngine(max_history=3)

    # --- SNIPPET 2: INITIALIZE THE DIRECTORY AND CV2 VIDEO WRITER INSTANCE ---
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print(f"[STATUS] Synchronization complete. Parsing {total_frames} total frames...")
    
    # Persistent dictionary tracking history state across frame boundaries
    active_tracks = {1: None, 2: None}
    csv_records = []
    frame_idx = 0

    while cap.isOpened() and d_cap.isOpened():
        ret, frame = cap.read()
        ret_d, d_frame = d_cap.read()
        
        if not frame_idx % 100:
            print(f" -> Processing Timeline Progress: {frame_idx} / {total_frames}")
        
        if not ret or not ret_d:
            break

        depth_gray = d_frame[:, :, 0]
        raw_hand_masks, object_masks = vision_eng.extract_masks(frame, width, height)
        
        # Let the dedicated Hand Tracking Engine do the heavy lifting
        tracked_hands, active_tracks, tracked_backgrounds = tracker.track_hand_identities(
            raw_hand_masks, width, height
        )

        # Baseline template configuration row structure
        record = {
            "Frame": frame_idx,
            "object_hand1": "None",
            "num_pixels_overlapping_hand1": 0,
            "average_depth_overlapping_pixels_hand1": 0.0,
            "depth_center_object_hand1": 0.0,
            "absolute_depth_diff_hand1": "0.0",
            "object_hand2": "None",
            "num_pixels_overlapping_hand2": 0,
            "average_depth_overlapping_pixels_hand2": 0.0,
            "depth_center_object_hand2": 0.0,
            "absolute_depth_diff_hand2": "0.0"
        }

        # Process entries for Hand 1 and Hand 2 slots independently
        for hand_id in [1, 2]:
            h_mask = tracked_hands.get(hand_id)
            if h_mask is None:
                continue  # This specific hand channel is missing or obscured in this frame
                
            dilated_h_mask = dilate_hand_mask(h_mask, kernel_size=(3, 3))

            # Extract interaction layer lists from processing block
            res_objs, res_overlaps, res_avgs, res_centers, res_diffs = evaluate_hand_object_interactions(
                dilated_h_mask, object_masks, depth_gray, width, height
            )

            # If arrays populated, serialize parameters to strings inside the tracking frame record
            if res_objs:
                record[f"object_hand{hand_id}"] = ", ".join(res_objs)
                record[f"num_pixels_overlapping_hand{hand_id}"] = ", ".join(res_overlaps)
                record[f"average_depth_overlapping_pixels_hand{hand_id}"] = ", ".join(res_avgs)
                record[f"depth_center_object_hand{hand_id}"] = ", ".join(res_centers)
                record[f"absolute_depth_diff_hand{hand_id}"] = ", ".join(res_diffs)

        # --- SNIPPET 3: LAYERED RE-COLORING ENGINE PLACED DIRECTLY BEFORE RECORDS SAVE ---
        canvas = frame.copy()
        overlay = frame.copy()

        # Parse string metrics generated for csv to cross-reference which targets are active
        interacting_objects_this_frame = set()
        for hid in [1, 2]:
            obj_string = record.get(f"object_hand{hid}", "None")
            if obj_string and obj_string != "None":
                for obj in obj_string.split(", "):
                    interacting_objects_this_frame.add(obj.strip())

        # 1. Paint All Object Masks (Orange = Passive, Yellow = Actively Interacting)
        for item in object_masks:
            if not item or len(item) == 0: 
                continue
            obj_name, o_mask = (item[0], item[1]) if isinstance(item[0], str) else (item[1], item[0])
            if np.sum(o_mask) == 0: 
                continue

            display_name = "cellphone" if obj_name == "cell phone" else obj_name

            if display_name in interacting_objects_this_frame:
                color = (0, 255, 255)  # BGR YELLOW
                label_text = f"[ACTIVE] {display_name}"
            else:
                color = (255,0,255 )  # BGR MAGENTA
                label_text = display_name

            overlay[o_mask == 1] = color

            # Apply labels exactly over object contours
            contours, _ = cv2.findContours(o_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    ocx = int(M["m10"] / M["m00"])
                    ocy = int(M["m01"] / M["m00"])
                    cv2.putText(canvas, label_text, (ocx - 20, ocy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Paint Extra Background Hands (Green)
        for bg_id, bg_mask in tracked_backgrounds.items():
            overlay[bg_mask == 1] = (0, 255, 0)  
            y_ind, x_ind = np.where(bg_mask == 1)
            if len(x_ind) > 0:
                bg_cx, bg_cy = int(np.mean(x_ind)), int(np.mean(y_ind))
                cv2.putText(canvas, f"HAND {bg_id}", (bg_cx - 20, bg_cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Paint Primary Track 1 (Blue)
        if tracked_hands.get(1) is not None:
            overlay[tracked_hands[1] == 1] = (255, 0, 0)  # Blue
            if active_tracks[1] is not None:
                cx, cy = active_tracks[1]
                cv2.putText(canvas, "HAND 1", (cx - 30, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Paint Primary Track 2 (Red)
        if tracked_hands.get(2) is not None:
            overlay[tracked_hands[2] == 1] = (0, 0, 255)  # Red
            if active_tracks[2] is not None:
                cx, cy = active_tracks[2]
                cv2.putText(canvas, "HAND 2", (cx - 30, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Blend masked metrics with standard color distributions
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        # Top Information Dashboard Frame Panel Header
        cv2.rectangle(canvas, (0, 0), (width, 45), (0, 0, 0), -1)
        h1_str = "Active" if active_tracks[1] else ("Coasting" if tracked_hands.get(1) is not None else "Dead")
        h2_str = "Active" if active_tracks[2] else ("Coasting" if tracked_hands.get(2) is not None else "Dead")
        cv2.putText(canvas, f"FRAME: {frame_idx} | H1: {h1_str} | H2: {h2_str} | Active Interactions: {len(interacting_objects_this_frame)}", 
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Write out compiled visualization matrix frame
        out_video.write(canvas)

        csv_records.append(record)
        frame_idx += 1

    # Cleanup open system streaming interfaces
    cap.release()
    d_cap.release()
    # --- SNIPPET 4: CLOSE SECTIONS STREAM POOL ---
    out_video.release()

    # Commit structured output parameters to target workspace file path
    save_telemetry_report(csv_records, ORDERED_COLUMNS, CSV_OUTPUT_PATH)


if __name__ == "__main__":
    main()