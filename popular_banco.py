import os
import time # Adicionado para controlar o tempo de pausa
import pandas as pd
import chromadb
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

# 1. Carregando a chave da API
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class CustomGoogleEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = "models/gemini-embedding-2" # O modelo recomendado para embeddings

    def __call__(self, input: Documents) -> Embeddings:
        resultado = genai.embed_content(
            model=self.model,
            content=input,
            task_type="retrieval_document"
        )
        emb = resultado['embedding']
        if isinstance(emb[0], float):
            return [emb]
        return emb

minha_funcao_embedding = CustomGoogleEmbeddingFunction(api_key=GOOGLE_API_KEY)

# 2. Inicializa o banco de dados localmente
chroma_client = chromadb.PersistentClient(path="./chroma_data")

# 3. Cria a coleção usando a nossa função customizada
colecao_vendas = chroma_client.get_or_create_collection(
    name="walmart_vendas",
    embedding_function=minha_funcao_embedding
)

print("Lendo o arquivo CSV...")
df = pd.read_csv('dataset_walmart_rag_ptbr.csv')

documentos = df['texto_para_embedding'].tolist()
ids = [str(i) for i in range(len(df))]
metadados = [{"origem": "dataset_walmart"} for _ in range(len(df))]

# --- A MÁGICA DO THROTTLING AQUI ---
print("Gerando embeddings em lotes (batches) para evitar Rate Limit...")

tamanho_lote = 20 # Processa 20 registros por vez
total_registros = len(documentos)

for i in range(0, total_registros, tamanho_lote):
    fim = min(i + tamanho_lote, total_registros)
    print(f"Enviando lote de {i} até {fim-1} para a API do Google...")
    
    # Adiciona apenas a fatia atual (lote) no banco
    colecao_vendas.add(
        documents=documentos[i:fim],
        metadatas=metadados[i:fim],
        ids=ids[i:fim]
    )
    
    # Se ainda não for o último lote, faz uma pausa de 10 segundos
    if fim < total_registros:
        print("Aguardando 10 segundos para não estourar a cota da API...")
        time.sleep(50)

print(f"Sucesso! {colecao_vendas.count()} registros foram gravados fisicamente no banco vetorial.")