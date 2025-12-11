import os

user = "phd2"

if user == "phd2":
    user_base_path = "/home/phd2"
    base_data_path = "/home/phd2/Documenti/embryo/valencia"
    

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
