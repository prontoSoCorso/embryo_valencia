import os

user = "phd2"

if user == "phd2":
    user_base_path = "/home/phd2"
    base_data_path = "/home/phd2/Documenti/embryo/marilena_videos"
    

class Config_preprocessing:
    base_data_path = base_data_path
    
    # Input PDB files directory
    input_dir_pdb_files = os.path.join(base_data_path, "pdb_files")

    # Output directory for extracted images from PDB files
    output_dir_extracted_pdb_files = os.path.join(base_data_path, "extracted_data")

    # Log file for PDB extraction process
    log_file_pdb_extraction = os.path.join(base_data_path, "logs/pdb_extraction_log.txt")

    # Source directory for extracted equatorial images
    src_dir_extracted_pdb = output_dir_extracted_pdb_files

    # Destination directory for equatorial images
    dest_dir_extracted_equator = os.path.join(base_data_path, "extracted_equatorial_frames")

    # Log file for extracting equatorial frames
    log_file_equatorial_extraction = os.path.join(base_data_path, "logs/equatorial_extraction_log.txt")

    # rejected_dir for quality check
    rejected_dir_quality_check = os.path.join(base_data_path, "rejected_equator_videos")

    # Log file for quality check process
    log_file_quality_check = os.path.join(base_data_path, "logs/quality_check_log.txt")

    # Path for checking final folder structure
    path_main_folder = dest_dir_extracted_equator

    # Valid wells output file
    valid_wells_file = os.path.join(base_data_path, "valid_wells/valid_wells.txt")

    # log file for stats timing
    log_file_stats_timing = os.path.join(base_data_path, "logs/stats_timing_log.txt")


class Config_OpticalFlow:
    base_project_path = "/home/phd2/Scrivania/CorsoRepo/embryo_valencia"
    base_project_database_path = os.path.join(base_project_path, "datasets")
    base_project_optical_flow = os.path.join(base_project_path, "_01_opticalFlows")

    # ---------------- PATHS ----------------
    # Input directory (where your sorted videos are)
    input_video_dir = os.path.join(base_data_path, "final_videos")
    
    # Base output directory for metrics and images
    output_base_dir = os.path.join(base_project_optical_flow, "optical_flow_results")
    
    # Pickle output directory (for temporal data storage)
    pickle_dir = os.path.join(base_project_database_path, "pickles")
    
    # Image output directory (for overlays)
    output_path_images = os.path.join(base_data_path, "optical_flow_images")
    
    # Metrics plots output directory
    output_path_metrics = os.path.join(output_base_dir, "metrics_plots")

    # ---------------- PROCESSING PARAMETERS ----------------
    method_optical_flow = "Farneback"
    img_size = 500                      # Resize frames to this square size
    
    # Frame trimming options
    num_minimum_frames = 40             # Minimum frames required to process a video
    num_initial_frames_to_cut = 0       # Skip unstable initial frames
    num_forward_frame = 3               # Window for sum_mean_mag calculation

    # Saving flags
    save_metrics = False                 # Save plots of metrics
    save_overlay_optical_flow = False    # Save frames with flow arrows
    save_final_data = True              # Save pickle files
    
    # ---------------- FARNEBACK PARAMETERS ----------------
    pyr_scale = 0.5
    levels = 4
    winSize_Farneback = 25
    iterations = 3
    poly_n = 5
    poly_sigma = 1.2
    flags = 0