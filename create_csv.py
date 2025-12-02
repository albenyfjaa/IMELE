import os
import glob
import pandas as pd

# ================= CONFIGURAÇÃO =================
# Pastas onde estão suas imagens CORTADAS (256x256)
# Ajuste estes caminhos para onde seus dados realmente estão
PASTA_RGB = r'D:\git\im2height\data\raw\test\images'     
PASTA_DEPTH = r'D:\git\im2height\data\raw\test\dsm' 

# Extensões (verifique se suas imagens são .jpg, .png ou .tif)
EXT_RGB = '*.tif'   # Ex: *.png, *.tif
EXT_DEPTH = '*.tif' # Ex: *.png, *.tif

# Nome do arquivo final
NOME_CSV = 'meu_treino.csv'
# ================================================

def formatar_caminho(caminho_absoluto):
    # Pega o caminho relativo a partir da pasta onde o script está rodando
    try:
        rel_path = os.path.relpath(caminho_absoluto, os.getcwd())
    except ValueError:
        # Fallback se estiver em drives diferentes
        rel_path = caminho_absoluto

    # Força barra normal (/) mesmo no Windows
    return rel_path.replace(os.sep, '/')

def gerar_csv():
    # 1. Encontrar arquivos (usando glob recursivo se precisar)
    # Se quiser buscar em subpastas, use recursive=True e '**/*.jpg'
    lista_rgb = sorted(glob.glob(os.path.join(PASTA_RGB, EXT_RGB)))
    lista_depth = sorted(glob.glob(os.path.join(PASTA_DEPTH, EXT_DEPTH)))

    print(f"Encontrados: {len(lista_rgb)} RGB e {len(lista_depth)} Depth")

    if len(lista_rgb) == 0:
        print("❌ ERRO: Nenhuma imagem encontrada! Verifique o caminho em PASTA_RGB.")
        return

    # 2. Parear arquivos
    dados = []
    
    # Cria um dicionário para achar o depth pelo nome rápido
    # Ex: 'imagem_01' -> 'caminho/completo/depth/imagem_01.png'
    dict_depth = {os.path.splitext(os.path.basename(f))[0]: f for f in lista_depth}

    for arq_rgb in lista_rgb:
        nome_base = os.path.splitext(os.path.basename(arq_rgb))[0]
        
        # Tenta achar o arquivo de depth com o MESMO nome
        if nome_base in dict_depth:
            arq_depth = dict_depth[nome_base]
            
            # Formata para o padrão do IMELE (barras / e relativo)
            caminho_rgb_formatado = formatar_caminho(arq_rgb)
            caminho_depth_formatado = formatar_caminho(arq_depth)
            
            dados.append([caminho_rgb_formatado, caminho_depth_formatado])
        else:
            print(f"⚠️ Aviso: Sem par de profundidade para {nome_base}")

    # 3. Salvar CSV
    df = pd.DataFrame(dados)
    # header=False remove o cabeçalho, index=False remove a numeração das linhas
    df.to_csv(NOME_CSV, header=False, index=False)
    
    print(f"\n✅ Arquivo '{NOME_CSV}' criado com sucesso!")
    print(f"Total de pares: {len(dados)}")
    print("Exemplo das primeiras linhas (verifique as barras):")
    print(df.head())

if __name__ == "__main__":
    gerar_csv()