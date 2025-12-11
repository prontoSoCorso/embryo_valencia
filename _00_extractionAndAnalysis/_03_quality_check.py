import os
import shutil

def check_and_filter_videos(src_dir, rejected_dir, log_file, min_frames=20):
    """
    Scans the source directory for video folders.
    Moves folders with < min_frames to a rejected directory.
    Logs the operations.
    """
    # Setup Log
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    else:
        log_file = "quality_check_log.txt"

    # Setup Rejected Dir
    os.makedirs(rejected_dir, exist_ok=True)

    stats = {
        "total_videos_checked": 0,
        "valid_videos": 0,
        "rejected_videos": 0,
        "rejection_reasons": {}
    }

    try:
        print("\n========================================")
        print("Starting quality check...")
        print("========================================\n")

        with open(log_file, "w") as log:
            log.write(f"Starting Quality Check on: {src_dir}\n")
            log.write(f"Minimum frames required: {min_frames}\n")
            log.write("-" * 50 + "\n")

            # Iterate Years
            for year in sorted(os.listdir(src_dir)):
                year_path = os.path.join(src_dir, year)
                
                if not os.path.isdir(year_path):
                    continue

                # Iterate Videos
                for video_folder in sorted(os.listdir(year_path)):
                    video_path = os.path.join(year_path, video_folder)
                    
                    if not os.path.isdir(video_path):
                        continue

                    stats["total_videos_checked"] += 1
                    
                    # Count valid images
                    frames = [f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    num_frames = len(frames)

                    # --- CHECK LOGIC ---
                    reject_reason = None
                    
                    if num_frames == 0:
                        reject_reason = "Empty Folder"
                    elif num_frames < min_frames:
                        reject_reason = f"Insufficient frames ({num_frames} < {min_frames})"

                    # --- ACTION ---
                    if reject_reason:
                        stats["rejected_videos"] += 1
                        
                        # Prepare destination in rejected folder (maintain year structure)
                        rej_year_path = os.path.join(rejected_dir, year)
                        os.makedirs(rej_year_path, exist_ok=True)
                        dest_path = os.path.join(rej_year_path, video_folder)

                        # Move folder
                        try:
                            # If it exists in rejected, delete it first to overwrite
                            if os.path.exists(dest_path):
                                shutil.rmtree(dest_path)
                            shutil.move(video_path, dest_path)
                            
                            log.write(f"[REJECTED] Year: {year} | Video: {video_folder} | Reason: {reject_reason}\n")
                        except Exception as e:
                            log.write(f"[ERROR] Could not move {video_folder}: {e}\n")
                    else:
                        stats["valid_videos"] += 1

            # Final Report
            log.write("-" * 50 + "\n")
            log.write("FINAL STATISTICS:\n")
            log.write(f"Total Videos Checked: {stats['total_videos_checked']}\n")
            log.write(f"Valid Videos Kept:    {stats['valid_videos']}\n")
            log.write(f"Rejected Videos:      {stats['rejected_videos']}\n")
            
            print(f"Quality check finished. Rejected {stats['rejected_videos']} videos. See log: {log_file}")

    except Exception as e:
        print(f"Critical error during quality check: {e}")