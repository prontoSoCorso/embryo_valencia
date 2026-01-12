import os
import sys

# Configurazione dei percorsi e dei parametri
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import Config_preprocessing as cfg

try:
    from _01_extract_pdb import extract_frames
    from _02_extract_equatore import copy_equatorial_frames
    from _03_quality_check import check_and_filter_videos
except ImportError:
    from _00_extractionAndAnalysis._01_extract_pdb import extract_frames
    from _00_extractionAndAnalysis._02_extract_equatore import copy_equatorial_frames
    from _00_extractionAndAnalysis._03_quality_check import check_and_filter_videos

def main(extract_pdb=True, extract_equator=True, quality_check=True):
    
    ##############################
    # Load Config
    ##############################
    input_dir_pdb_files = cfg.input_dir_pdb_files
    output_dir_extracted_pdb_files = cfg.output_dir_extracted_pdb_files
    log_file_pdb_extraction = cfg.log_file_pdb_extraction

    src_dir_extracted_pdb = cfg.src_dir_extracted_pdb
    dest_dir_extracted_equator = cfg.dest_dir_extracted_equator
    log_file_equatorial_extraction = cfg.log_file_equatorial_extraction
    
    rejected_dir = cfg.rejected_dir_quality_check
    log_file_quality = cfg.log_file_quality_check

    ##############################
    # 1. Extraction from pdb files
    ##############################
    if extract_pdb:
        print("\n--- STEP 1: PDB Extraction ---")
        extract_frames(input_dir=input_dir_pdb_files, 
                       output_dir=output_dir_extracted_pdb_files, 
                       log_file=log_file_pdb_extraction,
                       first_year=2016, last_year=2050)

    ##############################
    # 2. Selecting equatorial images
    ##############################
    if extract_equator:
        print("\n--- STEP 2: Equatorial Selection & Renaming ---")
        copy_equatorial_frames(src_dir=src_dir_extracted_pdb, 
                               dest_dir=dest_dir_extracted_equator, 
                               log_file=log_file_equatorial_extraction,
                               first_year=2016, last_year=2050,
                               save_in_hours=True)

    ##############################
    # 3. Quality Check & Filtering
    ##############################
    if quality_check:
        print("\n--- STEP 3: Quality Check & Filtering ---")
        # Filter out videos with less than 40 frames (too short for analysis)
        check_and_filter_videos(src_dir=dest_dir_extracted_equator, 
                                rejected_dir=rejected_dir, 
                                log_file=log_file_quality, 
                                min_frames=40)

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # You can toggle steps here
    main(extract_pdb=True, extract_equator=True, quality_check=True)
    
    print(f"\nTotal Execution time: {time.time()-start_time:.2f} seconds")