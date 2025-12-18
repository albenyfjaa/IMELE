import os
import glob
import pandas as pd

# ================= CONFIGURAÇÃO =================
# Ajuste os caminhos se necessário
PASTA_RGB = r'C:\Users\LAB IA\Documents\albeny\datasets\dfc2019\extraidos_v2\dividido_v2\validacao\RGB'     
PASTA_DEPTH = r'C:\Users\LAB IA\Documents\albeny\datasets\dfc2019\extraidos_v2\dividido_v2\validacao\Truth' 

# Extensões
EXT_RGB = '*.tif'   
EXT_DEPTH = '*.tif' 

# Nome do arquivo final
NOME_CSV = 'dfc2019_validacao.csv'
# ================================================

def formatar_caminho(caminho_absoluto):
    # Pega o caminho relativo a partir da pasta onde o script está rodando
    try:
        rel_path = os.path.relpath(caminho_absoluto, os.getcwd())
    except ValueError:
        rel_path = caminho_absoluto

    # Força barra normal (/) para compatibilidade com Linux/PyTorch
    return rel_path.replace(os.sep, '/')

def gerar_csv():
    # 1. Encontrar arquivos
    lista_rgb = sorted(glob.glob(os.path.join(PASTA_RGB, EXT_RGB)))
    lista_depth = sorted(glob.glob(os.path.join(PASTA_DEPTH, EXT_DEPTH)))

    print(f"Encontrados: {len(lista_rgb)} RGB e {len(lista_depth)} Depth")

    if len(lista_rgb) == 0:
        print("❌ ERRO: Nenhuma imagem encontrada! Verifique o caminho em PASTA_RGB.")
        return

    # 2. Parear arquivos
    dados = []
    
    # Cria um dicionário para achar o depth rápido
    dict_depth = {os.path.splitext(os.path.basename(f))[0]: f for f in lista_depth}

    for arq_rgb in lista_rgb:
        nome_base_rgb = os.path.splitext(os.path.basename(arq_rgb))[0]
        
        # --- CORREÇÃO AQUI ---
        # Transformamos o nome RGB no nome esperado do Depth (AGL)
        # Substitui "_RGB_" por "_AGL_"
        nome_esperado_depth = nome_base_rgb.replace('_RGB_', '_AGL_')
        # ---------------------

        # Procura usando o nome transformado (AGL)
        if nome_esperado_depth in dict_depth:
            arq_depth = dict_depth[nome_esperado_depth]
            
            caminho_rgb_formatado = formatar_caminho(arq_rgb)
            caminho_depth_formatado = formatar_caminho(arq_depth)
            
            dados.append([caminho_rgb_formatado, caminho_depth_formatado])
        else:
            print(f"⚠️ Aviso: Sem par (AGL) para {nome_base_rgb}")

    # 3. Salvar CSV
    if len(dados) > 0:
        df = pd.DataFrame(dados)
        df.to_csv(NOME_CSV, header=False, index=False)
        
        print(f"\n✅ Arquivo '{NOME_CSV}' criado com sucesso!")
        print(f"Total de pares criados: {len(dados)}")
        print("Exemplo das primeiras linhas:")
        print(df.head())
    else:
        print("\n❌ Nenhum par foi formado. Verifique se os nomes batem após a troca de RGB por AGL.")

if __name__ == "__main__":
    gerar_csv()