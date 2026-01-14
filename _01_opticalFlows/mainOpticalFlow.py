import os
import sys
import pickle
import logging

# Configurazione dei percorsi e dei parametri
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import Config_OpticalFlow as config
from _process_frames_of import process_frames

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("optical_flow_analysis.log")
        ]
    )
    
    # Setup Output Directories
    out_dirs = {
        'pickles': config.pickle_dir,
        'images': config.output_path_images,
        'metrics': config.output_path_metrics
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
        
    # Initialize global storage for final pickles
    # Added 'timings' to store the extracted hours
    all_data = {
        'mean_magnitude': {},
        'vorticity': {},
        'sum_mean_mag': {},
        'timings': {} 
    }
    
    # Walk through input directory
    input_root = config.input_video_dir
    logging.info(f"Scanning input directory: {input_root}")
    
    processed_count = 0
    
    if not os.path.exists(input_root):
        logging.error(f"Input directory not found: {input_root}")
        return

    for category in os.listdir(input_root):
        cat_path = os.path.join(input_root, category)
        if not os.path.isdir(cat_path):
            continue
            
        logging.info(f"Processing Category: {category}")
        
        # Iterate over Embryos in Category
        for embryo_id in os.listdir(cat_path):
            embryo_path = os.path.join(cat_path, embryo_id)
            if not os.path.isdir(embryo_path):
                continue
                
            logging.info(f"  Processing Embryo: {embryo_id}")
            
            try:
                metrics = process_frames(embryo_path, embryo_id, out_dirs)
                
                if metrics:
                    # Accumulate data into all_data dicts
                    for key in all_data:
                        if key in metrics:
                            all_data[key][embryo_id] = metrics[key]
                    processed_count += 1
                    
            except Exception as e:
                logging.error(f"Failed to process {embryo_id}: {e}")

    # --- SAVE FINAL PICKLES ---
    if config.save_final_data and processed_count > 0:
        logging.info("Saving final pickle files...")
        for key, data in all_data.items():
            fname = f"{key}.pkl"
            save_path = os.path.join(out_dirs['pickles'], fname)
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            logging.info(f"Saved {fname} (Contains {len(data)} entries)")

    logging.info(f"Analysis Complete. Processed {processed_count} videos.")

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    logging.info(f"Total execution time: {elapsed:.2f} seconds")
    # 1692 seconds for 168 videos (10 second per video on average)
    
