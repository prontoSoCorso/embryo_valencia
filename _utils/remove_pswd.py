import msoffcrypto

# --- CONFIGURAZIONE ---
input_path = "/home/phd2/Scrivania/CorsoRepo/embryo_valencia/DB MII to blasto.xlsx"
output_path = "/home/phd2/Scrivania/CorsoRepo/embryo_valencia/DB_MII_to_blasto_no_psw.xlsx"

# pswd
excel_password = ""

try:
    # Apre il file criptato in modalità lettura binaria
    encrypted_file = open(input_path, "rb")
    file = msoffcrypto.OfficeFile(encrypted_file)

    # Carica la chiave (password)
    file.load_key(password=excel_password)

    # Decripta e salva nel nuovo percorso in modalità scrittura binaria
    with open(output_path, "wb") as f:
        file.decrypt(f)

    print("✅ Successo! Il file decriptato è stato salvato in:")
    print(output_path)

    # Chiude il file originale
    encrypted_file.close()

except Exception as e:
    print(f"❌ Errore durante la decriptazione: {e}")