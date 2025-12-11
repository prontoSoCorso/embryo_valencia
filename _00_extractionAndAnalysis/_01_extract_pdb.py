import sqlite3
import os
import shutil

def write_to_file(data, filename):
    """Scrive i dati binari su disco."""
    with open(filename, 'wb') as file:
        file.write(data)

def extract_frames(input_dir, output_dir, log_file, first_year:int=1900, last_year:int=3000):
    """
    Extracts frames from SQLite database files.
    
    Args:
        input_dir: Path to directory containing year-organized PDB files
        output_dir: Path where extracted images should be stored
        log_file: Path for operation log file
        first_year: Minimum year to process (inclusive)
        last_year: Maximum year to process (inclusive)
    """
    metrics = {}  # dict to hold metrics per year
    sep = "_"

    # Log file
    if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
    else:
        log_file = "extraction_log.txt"

    try:
        for year in sorted(os.listdir(input_dir)):
            year_path = os.path.join(input_dir, year)

            # --- CONTROL ---
            # 1. Se non è una cartella, ignoralo (salta i file sparsi come .pdb-shm nella root)
            if not os.path.isdir(year_path):
                continue
            
            # 2. Se il nome della cartella non è composto solo da numeri, ignoralo
            if not year.isdigit():
                print(f"Skipping non-year directory: {year}")
                continue
            # -----------------------

            if (first_year<=int(year)<=last_year):
                print(f"========== Extracting year: {year} ==========")
                metrics[year] = {"videos_extracted": 0, "errors": 0}  # Initialize metrics for year
                output_year_dir = os.path.join(output_dir, year)
                os.makedirs(output_year_dir, exist_ok=True)

                for file in sorted(os.listdir(year_path)):
                    if file.endswith('.pdb'):
                        print(file)
                        pdb_file = os.path.join(year_path, file)
                        pdb_name = os.path.splitext(file)[0]

                        try:
                            # Calculate total space needed for this PDB
                            with sqlite3.connect(pdb_file) as temp_con:
                                cur = temp_con.cursor()
                                res = cur.execute('SELECT SUM(LENGTH(Image)) FROM IMAGES')
                                total_size = res.fetchone()[0] or 0
                            
                            # Check disk space
                            disk_stat = shutil.disk_usage(output_dir)
                            if disk_stat.free < total_size*1.05:
                                print(f"Space insufficient for {pdb_file}. Skipping.")
                                metrics[year]["errors"] += 1
                                with open(log_file, "a") as log:
                                    log.write(f"Insufficient space for {pdb_file}. Needed: {total_size}, Free: {disk_stat.free}\n")
                                break
                            
                            # If there is enough space, continue extracting
                            # Images extraction
                            with sqlite3.connect(pdb_file) as con:
                                cur = con.cursor()
                                res = cur.execute("SELECT * FROM IMAGES")
                                images = res.fetchall()

                                wells = {row[0] for row in images}

                                for well in wells:
                                    video_dir = os.path.join(output_year_dir, f"{pdb_name}{sep}{well}")
                                    os.makedirs(video_dir, exist_ok=True)
                                    metrics[year]["videos_extracted"] += 1  # Incrementa il conteggio video

                                for row in images:
                                    well_id = row[0]
                                    timestamp = f"{row[1]}_{row[2]}_{row[3]}"
                                    image_data = row[4]

                                    image_filename = f"{pdb_name}{sep}{well_id}{sep}{timestamp}.jpg"
                                    image_path = os.path.join(output_year_dir, f"{pdb_name}{sep}{well_id}", image_filename)

                                    write_to_file(image_data, image_path)

                            print(f"Extraction completed for {pdb_file}")

                        except Exception as e:
                            print(f"========== Error with file {pdb_file}: {str(e)} ==========")
                            metrics[year]["errors"] += 1  # Incrementa il conteggio errori

    except Exception as e:
        print(f"========== General error: {str(e)} ==========")

    # Salva i risultati in un file di log
    try:
        with open(log_file, "a") as log:
            for year, data in metrics.items():
                log.write(f"Year: {year}\n")
                log.write(f"  Videos extracted: {data['videos_extracted']}\n")
                log.write(f"  Errors: {data['errors']}\n\n")
        print(f"Results saved in {log_file}")
    except Exception as e:
        print(f"Error while saving log: {e}")
