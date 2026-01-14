import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def extract_timestamp_from_filename(filename):
    """
    Extracts the time in hours from filenames like:
    'D2016..._0.25h.jpg' -> 0.25
    """
    try:
        # Get the last part after the last underscore: "0.25h.jpg"
        last_part = filename.split('_')[-1]
        
        # Remove file extension (e.g., ".jpg") -> "0.25h"
        name_no_ext = os.path.splitext(last_part)[0]
        
        # Remove 'h' if present and convert to float
        if name_no_ext.endswith('h'):
            time_str = name_no_ext[:-1] # "0.25"
            return float(time_str)
        else:
            # Fallback if 'h' is missing but it's a number
            return float(name_no_ext)
    except Exception as e:
        # Return None or NaN if extraction fails
        return float('nan')

def sort_files_by_slice_number(file_list):
    """Sorts frames by extracting the slice number index from filenames."""
    def get_slice_number(filename):
        # Heuristic: Find the token that looks like a frame count
        parts = filename.split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0

    filtered_files = [f for f in file_list if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    sorted_files = sorted(filtered_files, key=get_slice_number)
    return sorted_files

def preprocess_frame(frame, target_size):
    """Resizes and preprocesses the frame (grayscale, ensures uint8)."""
    if frame is None:
        return None
        
    if frame.shape[:2] != (target_size, target_size):
        frame = cv2.resize(frame, (target_size, target_size))
    
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    return frame

def calculate_vorticity(flow):
    """Calculates the mean vorticity (curl) of the flow field."""
    grad = np.gradient(flow, axis=(0, 1))
    vorticity = np.mean(grad[1][..., 1] - grad[0][..., 0])
    return vorticity

def overlay_arrows(image, flow, step=16):
    """Draws sparse optical flow arrows on the image."""
    h, w = image.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # ARROW COLOR: BGR Format (Orange)
    arrow_color = (0, 140, 255) 
    
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), arrow_color, 1, tipLength=0.3)
    return vis

def save_plot_temporal_metrics(output_path, dish_well, metrics):
    """Saves a summary plot of all metrics with DISTINCT colors."""
    
    # Define distinct colors for specific metrics
    metric_colors = {
        'mean_magnitude': 'royalblue',
        'vorticity': 'forestgreen',
        'sum_mean_mag': 'darkorange'
    }
    default_color = 'gray'

    # Filter out 'timings' from plotting if present, as it's the x-axis, not a metric
    plot_metrics = {k: v for k, v in metrics.items() if k != 'timings'}

    fig, axes = plt.subplots(len(plot_metrics), 1, figsize=(10, 12))
    if len(plot_metrics) == 1: axes = [axes]
    
    for ax, (key, values) in zip(axes, plot_metrics.items()):
        color = metric_colors.get(key, default_color)
        
        ax.plot(values, label=key, color=color, linewidth=2)
        ax.set_title(key, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        ax.set_ylabel("Value")
        
        if ax == axes[-1]:
            ax.set_xlabel("Frame Index")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compute_optical_flow_farneback(prev_frame, curr_frame, config):
    """Wrapper for OpenCV Farneback calculation."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame, curr_frame, None,
        config.pyr_scale, 
        config.levels, 
        config.winSize_Farneback,
        config.iterations, 
        config.poly_n, 
        config.poly_sigma, 
        config.flags
    )
    return flow