import os
import sys
from pathlib import Path
import pandas as pd

# Clean, single-line project directory resolution
base_project_dir = Path(__file__).resolve().parent

# =====================================================================
# CONFIGURATION & FILE PATHS
# =====================================================================
FILENAME = "debug_interaction_hybrid_c18x1.csv"
GT_FILENAME = "C18.csv"
X_THRESHOLD = 2  # The maximum frame size parameter for gap filling/removal

INPUT_CSV_PATH = base_project_dir / "validation_checks" / FILENAME
GT_CSV_PATH = base_project_dir / "GT" / GT_FILENAME
OUTPUT_CSV_PATH = base_project_dir / "validation_checks" / f"postprocess_{FILENAME}"
EVAL_CSV_PATH = base_project_dir / "evaluation" / f"evaluation_report_{GT_FILENAME}"

# =====================================================================
# TEMPORAL CORE ENGINE SUB-ROUTINES
# =====================================================================

def parse_telemetry_to_timelines(df, hand_id):
    """Parses row strings to create independent object timelines."""
    obj_col = f"object_hand{hand_id}"
    ovr_col = f"num_pixels_overlapping_hand{hand_id}"
    avg_col = f"average_depth_overlapping_pixels_hand{hand_id}"
    ctr_col = f"depth_center_object_hand{hand_id}"
    dif_col = f"absolute_depth_diff_hand{hand_id}"
    
    total_frames = len(df)
    object_timelines = {}

    for t in range(total_frames):
        o_val = str(df.at[t, obj_col]).strip()
        if o_val in ["None", "", "0"]: continue
            
        objs = [o.strip() for o in o_val.split(",")]
        ovrs = [v.strip() for v in str(df.at[t, ovr_col]).split(",")]
        avgs = [v.strip() for v in str(df.at[t, avg_col]).split(",")]
        ctrs = [v.strip() for v in str(df.at[t, ctr_col]).split(",")]
        difs = [v.strip() for v in str(df.at[t, dif_col]).split(",")]
        
        for i, obj_name in enumerate(objs):
            if obj_name not in object_timelines:
                object_timelines[obj_name] = [None] * total_frames
                
            object_timelines[obj_name][t] = {
                "overlap": ovrs[i] if i < len(ovrs) else "0",
                "avg_depth": avgs[i] if i < len(avgs) else "0.0",
                "ctr_depth": ctrs[i] if i < len(ctrs) else "0.0",
                "diff_str": difs[i] if i < len(difs) else "0.0"
            }
    return object_timelines


def apply_temporal_filters(object_timelines, total_frames, x_limit):
    """Fills gaps <= X (with diff=0.0) and removes short isolated spikes <= X."""
    filtered_timelines = {}

    for obj_name, timeline in object_timelines.items():
        new_timeline = list(timeline)
        
        # --- PHASE 1: FILL FALSE NEGATIVE GAPS ---
        last_seen_idx = None
        for t in range(total_frames):
            if new_timeline[t] is not None:
                if last_seen_idx is not None:
                    gap_size = t - last_seen_idx - 1
                    if 0 < gap_size <= x_limit:
                        for fill_t in range(last_seen_idx + 1, t):
                            new_timeline[fill_t] = {
                                "overlap": "0", "avg_depth": "0.0", "ctr_depth": "0.0", "diff_str": "0.0"
                            }
                last_seen_idx = t

        # --- PHASE 2: WIPE FALSE POSITIVE SPIKES ---
        t = 0
        while t < total_frames:
            if new_timeline[t] is not None:
                start_block = t
                while t < total_frames and new_timeline[t] is not None: t += 1
                end_block = t
                if (end_block - start_block) <= x_limit:
                    for wipe_t in range(start_block, end_block): new_timeline[wipe_t] = None
            else:
                t += 1
                
        if any(item is not None for item in new_timeline):
            filtered_timelines[obj_name] = new_timeline
    return filtered_timelines


def resolve_one_object_per_frame(filtered_timelines, total_frames):
    """If multiple objects exist in frame t, only the one with the lowest depth delta survives."""
    for t in range(total_frames):
        active_candidates = []
        for obj_name, timeline in filtered_timelines.items():
            if timeline[t] is not None:
                try:
                    diff_val = float(timeline[t]["diff_str"])
                except ValueError:
                    diff_val = float('inf')
                active_candidates.append((obj_name, diff_val))
                
        if len(active_candidates) > 1:
            active_candidates.sort(key=lambda x: x[1])
            winner_name = active_candidates[0][0]
            for obj_name, timeline in filtered_timelines.items():
                if obj_name != winner_name:
                    timeline[t] = None
    return filtered_timelines


def apply_identity_lock_pass(filtered_timelines, total_frames):
    """Corrects identity swap anomalies at gap-fill boundaries."""
    for t in range(1, total_frames):
        prev_obj = None
        for obj_name, timeline in filtered_timelines.items():
            if timeline[t-1] is not None:
                prev_obj = obj_name
                break
        if prev_obj is None: continue
            
        current_obj = None
        current_data = None
        for obj_name, timeline in filtered_timelines.items():
            if timeline[t] is not None:
                current_obj = obj_name
                current_data = timeline[t]
                break
        if current_obj is None: continue
            
        if current_obj != prev_obj and current_data["diff_str"] == "0.0":
            filtered_timelines[prev_obj][t] = current_data
            filtered_timelines[current_obj][t] = None
    return filtered_timelines


def rebuild_dataframe_columns(df, filtered_timelines, total_frames, hand_id):
    """Reassembles individual object vectors back into unified dataframe rows."""
    obj_col = f"object_hand{hand_id}"
    ovr_col = f"num_pixels_overlapping_hand{hand_id}"
    avg_col = f"average_depth_overlapping_pixels_hand{hand_id}"
    ctr_col = f"depth_center_object_hand{hand_id}"
    dif_col = f"absolute_depth_diff_hand{hand_id}"

    for t in range(total_frames):
        row_objs, row_ovrs, row_avgs, row_ctrs, row_difs = [], [], [], [], []
        
        for obj_name, timeline in filtered_timelines.items():
            if timeline[t] is not None:
                data = timeline[t]
                row_objs.append(obj_name)
                row_ovrs.append(data["overlap"])
                row_avgs.append(data["avg_depth"])
                row_ctrs.append(data["ctr_depth"])
                row_difs.append(data["diff_str"])
                
        if row_objs:
            df.at[t, obj_col] = ", ".join(row_objs)
            df.at[t, ovr_col] = ", ".join(row_ovrs)
            df.at[t, avg_col] = ", ".join(row_avgs)
            df.at[t, ctr_col] = ", ".join(row_ctrs)
            df.at[t, dif_col] = ", ".join(row_difs)
        else:
            df.at[t, obj_col], df.at[t, ovr_col] = "None", "0"
            df.at[t, avg_col], df.at[t, ctr_col], df.at[t, dif_col] = "0.0", "0.0", "0.0"
    return df

# =====================================================================
# EVALUATION IMPLEMENTATION MODULE
# =====================================================================

def clean_text_value(val):
    """Standardizes empty baseline strings cleanly across nan types."""
    s = str(val).strip().lower()
    if s in ["none", "nan", "0", "", "null"]:
        return "none"
    return s


def run_ground_truth_evaluation(pred_df, gt_df):
    """Executes frame-by-frame subset analysis via strict explicit key mapping."""
    total_frames = len(pred_df)
    
    eval_metrics = {
        "GT_hand1": [], "GT_hand2": [],
        "TP_hand1": [], "TN_hand1": [], "FP_hand1": [], "FN_hand1": [],
        "TP_hand2": [], "TN_hand2": [], "FP_hand2": [], "FN_hand2": []
    }

    # Clean header column labels to ensure name checks resolve perfectly
    gt_df.columns = [str(c).strip().lower() for c in gt_df.columns]
    
    # Extract the exact column text designations
    frame_col_name = gt_df.columns[0]
    left_col_name = gt_df.columns[1]   # maps to hand1
    right_col_name = gt_df.columns[2]  # maps to hand2

    # Standardize lookup frame entries
    gt_df['frame_lookup_key'] = gt_df[frame_col_name].apply(clean_text_value)
    gt_df = gt_df.set_index('frame_lookup_key')

    for t in range(total_frames):
        frame_str = str(t)
        
        if frame_str in gt_df.index:
            gt_row = gt_df.loc[frame_str]
            if isinstance(gt_row, pd.DataFrame):
                gt_row = gt_row.iloc[0]
            
            # --- THE FIX: Lookup explicitly by column names, not positional integers ---
            gt_h1 = clean_text_value(gt_row[left_col_name])
            gt_h2 = clean_text_value(gt_row[right_col_name])
        else:
            gt_h1, gt_h2 = "none", "none"

        eval_metrics["GT_hand1"].append(gt_h1)
        eval_metrics["GT_hand2"].append(gt_h2)

        for hand_id, gt_val in [(1, gt_h1), (2, gt_h2)]:
            pred_raw = str(pred_df.at[t, f"object_hand{hand_id}"])
            pred_list = [clean_text_value(o) for o in pred_raw.split(",")]
            
            tp, tn, fp, fn = 0, 0, 0, 0

            if gt_val == "none":
                if len(pred_list) == 1 and pred_list[0] == "none":
                    tn = 1
                else:
                    fp = 1
            else:
                if gt_val in pred_list:
                    tp = 1
                else:
                    fn = 1

            eval_metrics[f"TP_hand{hand_id}"].append(tp)
            eval_metrics[f"TN_hand{hand_id}"].append(tn)
            eval_metrics[f"FP_hand{hand_id}"].append(fp)
            eval_metrics[f"FN_hand{hand_id}"].append(fn)

    for col_name, data_vector in eval_metrics.items():
        pred_df[col_name] = data_vector
        
    return pred_df


# =====================================================================
# MAIN RUNTIME ORCHESTRATION PIPELINE
# =====================================================================

def main():
    print("=" * 85)
    print("LAUNCHING CLEAN EVALUATION & OBJECT-AWARE TELEMETRY FILTER PIPELINE")
    print(f" -> Configured Frame Threshold Window (X): {X_THRESHOLD}")
    print("=" * 85)

    if not INPUT_CSV_PATH.exists():
        print(f"[-] Error: Target input telemetry file could not be found.\nChecked: {INPUT_CSV_PATH}")
        sys.exit()

    df = pd.read_csv(INPUT_CSV_PATH, sep=";", dtype=str)
    total_frames = len(df)

    for hand_id in [1, 2]:
        print(f" -> Processing Hand {hand_id} pass...")
        
        # Step 1: Split into unique object timelines
        timelines = parse_telemetry_to_timelines(df, hand_id)
        
        # Step 2: Apply temporal gap fill and spike wipe filters (Fill -> Erase)
        filtered_timelines = apply_temporal_filters(timelines, total_frames, X_THRESHOLD)
        
        # Step 3 & 4: Bypassed filters per design choice
        # resolved_timelines = resolve_one_object_per_frame(filtered_timelines, total_frames)
        # locked_timelines = apply_identity_lock_pass(resolved_timelines, total_frames)
        
        # Step 5: FIXED - Re-serialize back to dataframe structure using filtered timelines
        df = rebuild_dataframe_columns(df, filtered_timelines, total_frames, hand_id)

    # Save out the intermediate postprocessed tracking data file
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, sep=";", index=False)
    print(f"[SUCCESS] Cleaned single-object telemetry file exported to:\n -> {OUTPUT_CSV_PATH}\n")

    # --- PROCESS 2: RUN MUTUALLY-EXCLUSIVE ACCURACY EVALUATION ---
    print("[STATUS] Initializing statistical evaluation metrics pass against ground truth...")
    if not GT_CSV_PATH.exists():
        print(f"[-] Error: Ground Truth file could not be found.\nChecked: {GT_CSV_PATH}")
        sys.exit()

    gt_df = pd.read_csv(GT_CSV_PATH, sep=";", dtype=str)
    
    # Run fixed matrix evaluation loop
    evaluated_df = run_ground_truth_evaluation(df, gt_df)

    # Export report to evaluation directory path
    EVAL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    evaluated_df.to_csv(EVAL_CSV_PATH, sep=";", index=False)
    print(f"[SUCCESS] Statistical Evaluation Complete.")
    print(f" -> Combined accuracy report compiled at:\n -> {EVAL_CSV_PATH}\n")

if __name__ == "__main__":
    main()