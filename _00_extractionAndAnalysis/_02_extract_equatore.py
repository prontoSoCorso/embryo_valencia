import os
import shutil
import re
import time

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    (e.g. "Image_2.jpg" comes before "Image_10.jpg")
    """
    def atoi(t):
        return int(t) if t.isdigit() else t
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def copy_equatorial_frames(src_dir, dest_dir, log_file, first_year:int=1900, last_year:int=3000, save_in_hours:bool=True):
    """
    Copy only equatorial frames (Z=0) by analyzing the file name structure.
    Expected structure: ..._Index_Z_Timestamp.jpg
    
    Args:
        save_in_hours (bool): If True, renames the output files appending the relative time 
                              in hours (e.g., _0.00h.jpg, _0.25h.jpg) relative to the first frame.
    """
    # Log file setup
    if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
    else:
        log_file = "extraction_log.txt"
    
    number_of_copied_files = 0
    
    try:
        # Iterate through all years
        print("\n========================================")
        print("Starting equatorial frame selection...")
        print("========================================\n")
        for year in sorted(os.listdir(src_dir)):
            year_path = os.path.join(src_dir, year)

            # --- CONTROL ---
            if not os.path.isdir(year_path):
                continue
            if not year.isdigit():
                continue
            if not (first_year <= int(year) <= last_year):
                continue
            # -----------------------
        
            dest_year_path = os.path.join(dest_dir, year)
            os.makedirs(dest_year_path, exist_ok=True)

            # Iterate through the video subfolders
            for video_folder in sorted(os.listdir(year_path)):
                video_folder_path = os.path.join(year_path, video_folder)
                if not os.path.isdir(video_folder_path):
                    continue
                    
                dest_video_folder_path = os.path.join(dest_year_path, video_folder)
                os.makedirs(dest_video_folder_path, exist_ok=True)

                # 1. COLLECT VALID FRAMES FIRST (Buffer)
                valid_frames = []
                
                for file in os.listdir(video_folder_path):
                    if not file.endswith((".jpg", ".jpeg", ".png")):
                        continue

                    # Structure check: ..._Index_Z_Timestamp.jpg
                    parts = file.split('_')
                    
                    # Check if Z (penultimate part) is "0"
                    if len(parts) >= 2 and parts[-2] == "0":
                        valid_frames.append(file)

                # 2. SORT FRAMES NATURALLY
                # This ensures frame _2_ comes before _10_, and establishes the correct time sequence
                valid_frames.sort(key=natural_keys)

                if not valid_frames:
                    continue

                # 3. DETERMINE T0 (Start Time)
                t0 = 0.0
                if save_in_hours:
                    try:
                        # Extract timestamp from the first file in the sorted list
                        # Filename: ..._Z_Timestamp.jpg -> Split by '_' -> Last part is Timestamp.jpg -> Splitext
                        first_file = valid_frames[0]
                        t0_str = os.path.splitext(first_file.split('_')[-1])[0]
                        t0 = float(t0_str)
                    except Exception as e:
                        print(f"Warning: Could not parse timestamp for video {video_folder}. Disabling hour conversion. Error: {e}")
                        save_in_hours = False # Fallback for this video

                # 4. COPY AND RENAME
                for file in valid_frames:
                    src_file_path = os.path.join(video_folder_path, file)
                    
                    if save_in_hours:
                        try:
                            # Calculate current time
                            t_curr_str = os.path.splitext(file.split('_')[-1])[0]
                            t_curr = float(t_curr_str)
                            
                            # Calculate relative hours: (Days_diff) * 24
                            hours = (t_curr - t0) * 24.0

                            # Create new filename: Original name without original timestamp + _{hours}h + extension
                            name_no_ext, ext = os.path.splitext(file)
                            name_without_timestamp = '_'.join(name_no_ext.split('_')[:-1])
                            new_filename = f"{name_without_timestamp}_{hours:.2f}h{ext}"

                            dest_file_path = os.path.join(dest_video_folder_path, new_filename)
                        except:
                            # Fallback if parsing fails
                            dest_file_path = os.path.join(dest_video_folder_path, file)
                    else:
                        # Standard copy without renaming
                        dest_file_path = os.path.join(dest_video_folder_path, file)

                    shutil.copy2(src_file_path, dest_file_path)
                number_of_copied_files += 1

        # Summary Log
        print(f"Selection completed in {dest_dir}")
        print(f"Number of copied files: {number_of_copied_files}")
        with open(log_file, "a") as log:
            log.write("\n========================================\n")
            log.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"Selection completed in {dest_dir}\n")
            log.write(f"First year: {first_year}, Last year: {last_year}\n")
            log.write(f"Converted to relative hours: {save_in_hours}\n")
            log.write(f"Number of copied files: {number_of_copied_files}\n")

    except Exception as e:
        print(f"Critical error: {e}")