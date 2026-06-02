import os
import time
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone

# 1. Carregando as variáveis de ambiente
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 2. Inicializa o cliente do Pinecone e conecta ao Index
print("Conectando ao Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "vendas-walmart" # Nome exato do index que você criou no painel
index = pc.Index(index_name)

# 3. Lendo o arquivo CSV
print("Lendo o arquivo CSV...")
df = pd.read_csv('dataset_walmart_rag_ptbr.csv')

documentos = df['texto_para_embedding'].tolist()
total_registros = len(documentos)

print(f"Total de registros a processar: {total_registros}")
print("Gerando embeddings em lotes (batches) via Llama e salvando no Pinecone...")

# --- A MÁGICA DO THROTTLING MANTIDA ---
tamanho_lote = 20 # Processa 20 registros por vez

for i in range(0, total_registros, tamanho_lote):
    fim = min(i + tamanho_lote, total_registros)
    print(f"Processando lote de {i} até {fim-1}...")
    
    # 1. Pega apenas a fatia de textos do lote atual
    lote_textos = documentos[i:fim]
    
    # 2. Envia os textos para a API do Pinecone transformar em vetores usando o Llama
    # 'passage' é o tipo de input recomendado para dados que serão armazenados no banco
    resposta_embedding = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=lote_textos,
        parameters={"input_type": "passage"}
    )
    
    # 3. Prepara os dados no formato exato que o Pinecone exige para salvar (upsert)
    registros_pinecone = []
    
    for indice_local, embedding_obj in enumerate(resposta_embedding):
        indice_global = i + indice_local
        
        # Estrutura obrigatória: id, values (o vetor) e metadata (o texto original)
        registro = {
            "id": str(indice_global),
            "values": embedding_obj.values,
            "metadata": {
                "texto": lote_textos[indice_local],
                "origem": "dataset_walmart"
            }
        }
        registros_pinecone.append(registro)
    
    # 4. Grava o lote fisicamente no banco vetorial
    index.upsert(vectors=registros_pinecone)
    
    # 5. Pausa de segurança para não estourar a cota da API do Pinecone
    if fim < total_registros:
        print("Aguardando 10 segundos para não estourar a cota da API...")
        time.sleep(10) # Você pode ajustar para 50 se o Pinecone pedir, mas 10 geralmente basta

print(f"Sucesso! Os dados foram gravados fisicamente no Pinecone.")