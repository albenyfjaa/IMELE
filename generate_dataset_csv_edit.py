import os
import glob
import pandas as pd

# --- CONFIGURAÇÃO ---
# Seus caminhos exatos (usei r'' para evitar erro com as barras invertidas do Windows)
folder_rgb = r"D:\pged\DISSETERTACAO\dados\_treinamentos\_datasets\dfc2019\extraido\completo_512\_test_demo_512\rgb"
folder_dsm = r"D:\pged\DISSETERTACAO\dados\_treinamentos\_datasets\dfc2019\extraido\completo_512\_test_demo_512\dsm"

output_csv = "test_dataset.csv"
# --------------------

print(f"Buscando arquivos em:\nRGB: {folder_rgb}\nDSM: {folder_dsm}")

# Lista todos os arquivos (suporta tif, png, jpg)
extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg']
rgb_files = []
for ext in extensions:
    rgb_files.extend(glob.glob(os.path.join(folder_rgb, ext)))

print(f"Encontrados {len(rgb_files)} arquivos RGB.")

csv_data = []
found_count = 0
missing_count = 0

for rgb_path in rgb_files:
    # 1. Pega o nome do arquivo (ex: JAX_004_006_RGB_r0c0.tif)
    filename = os.path.basename(rgb_path)
    
    # 2. TRUQUE: Substitui _RGB_ por _AGL_ para achar o par correspondente
    # Isso resolve o problema de nomes diferentes
    dsm_filename = filename.replace('_RGB_', '_AGL_')
    
    # 3. Monta o caminho completo esperado do DSM
    dsm_path = os.path.join(folder_dsm, dsm_filename)
    
    # 4. Verifica se o arquivo DSM realmente existe
    if os.path.exists(dsm_path):
        # Adiciona ao CSV (formato: caminho_rgb;caminho_dsm)
        csv_data.append(f"{rgb_path};{dsm_path}")
        found_count += 1
    else:
        # Tenta procurar com outras extensões caso o DSM seja .tif e o RGB .jpg
        base_dsm = os.path.splitext(dsm_filename)[0] # remove extensão
        found_alt = False
        for ext in extensions:
            alt_dsm_path = os.path.join(folder_dsm, base_dsm + ext[1:]) # reconstrói nome
            if os.path.exists(alt_dsm_path):
                csv_data.append(f"{rgb_path};{alt_dsm_path}")
                found_count += 1
                found_alt = True
                break
        
        if not found_alt:
            print(f"ALERTA: Par não encontrado para {filename} (Esperado: {dsm_filename})")
            missing_count += 1

# Salva o arquivo CSV
if csv_data:
    with open(output_csv, "w") as f:
        f.write("\n".join(csv_data))
    print(f"\nSUCESSO! Arquivo '{output_csv}' gerado.")
    print(f"Pares encontrados: {found_count}")
    print(f"Pares perdidos: {missing_count}")
else:
    print("\nERRO: Nenhum par foi encontrado. Verifique os caminhos e os nomes.")