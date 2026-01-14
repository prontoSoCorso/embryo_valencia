import os
import cv2
import numpy as np
import logging
from config import Config_OpticalFlow as config
from _opticalFlow_functions import (
    sort_files_by_slice_number,
    preprocess_frame,
    calculate_vorticity,
    overlay_arrows,
    save_plot_temporal_metrics,
    compute_optical_flow_farneback,
    extract_timestamp_from_filename
)

def process_frames(folder_path, dish_well, output_dirs):
    """Processes a single video folder and returns metrics using Farneback."""
    
    files = sort_files_by_slice_number(os.listdir(folder_path))
    
    if len(files) < config.num_minimum_frames:
        logging.warning(f"Skipping {dish_well}: Not enough frames ({len(files)} < {config.num_minimum_frames})")
        return None

    # Skip initial frames
    files = files[config.num_initial_frames_to_cut:]
    
    # Read First Frame
    first_frame_path = os.path.join(folder_path, files[0])
    prev_frame = cv2.imread(first_frame_path, cv2.IMREAD_GRAYSCALE)
    
    if prev_frame is None:
        logging.error(f"Could not read first frame: {first_frame_path}. Skipping video.")
        return None
        
    prev_frame = preprocess_frame(prev_frame, config.img_size)
    
    metrics = {
        'mean_magnitude': [],
        'vorticity': [],
        'sum_mean_mag': [],
        'timings': []  # <-- Added container for timestamps
    }
    
    # Create specific output dir for images if needed
    img_out_dir = os.path.join(output_dirs['images'], dish_well)
    if config.save_overlay_optical_flow:
        os.makedirs(img_out_dir, exist_ok=True)

    # Process subsequent frames
    for i, fname in enumerate(files[1:], 1):
        curr_frame_path = os.path.join(folder_path, fname)
        curr_frame = cv2.imread(curr_frame_path, cv2.IMREAD_GRAYSCALE)
        
        if curr_frame is None:
            logging.warning(f"Skipping corrupt frame {fname} in {dish_well}")
            continue

        curr_frame = preprocess_frame(curr_frame, config.img_size)
        
        # --- FARNEBACK CALCULATION ---
        flow = compute_optical_flow_farneback(prev_frame, curr_frame, config)
        
        # --- METRICS ---
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        metrics['mean_magnitude'].append(np.mean(magnitude))
        metrics['vorticity'].append(calculate_vorticity(flow))
        
        # --- EXTRACT TIMING ---
        # Extract time from the *current* frame filename
        t_val = extract_timestamp_from_filename(fname)
        metrics['timings'].append(t_val)
        
        # --- VISUALIZATION ---
        if config.save_overlay_optical_flow:
            vis = overlay_arrows(curr_frame, flow)
            
            # Use ORIGINAL filename (swap extension to .png)
            base_name = os.path.splitext(fname)[0]
            save_name = f"{base_name}.png"
            
            cv2.imwrite(os.path.join(img_out_dir, save_name), vis)
            
        prev_frame = curr_frame

    # --- POST-PROCESSING METRICS ---
    mean_mag = np.array(metrics['mean_magnitude'])
    
    if len(mean_mag) == 0:
        logging.warning(f"No metrics calculated for {dish_well} (all frames might be corrupt).")
        return None

    sum_mean = []
    window = config.num_forward_frame
    for i in range(len(mean_mag) - window + 1):
        sum_mean.append(np.sum(mean_mag[i:i+window]))
    
    padding = len(mean_mag) - len(sum_mean)
    sum_mean = np.concatenate([sum_mean, np.zeros(padding)])
    metrics['sum_mean_mag'] = sum_mean.tolist()

    if config.save_metrics:
        plot_path = os.path.join(output_dirs['metrics'], f"{dish_well}_metrics.png")
        save_plot_temporal_metrics(plot_path, dish_well, metrics)
        
    return metrics