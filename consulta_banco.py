import chromadb

print("Conectando ao banco de dados vetorial...")

# 1. Conecta na pasta física onde os dados foram salvos
chroma_client = chromadb.PersistentClient(path="./chroma_data")

# 2. Acessa a coleção (tabela)
# Nota: Não precisamos carregar a API Key do Google aqui, pois não vamos gerar novos vetores, 
# apenas ler o que já está escrito no disco.
colecao_vendas = chroma_client.get_collection(name="walmart_vendas")

# 3. Conta o total exato de linhas
total_registros = colecao_vendas.count()
print(f"\nTotal de registros físicos no banco: {total_registros}")

# 4. Puxa uma amostra (ex: os 3 primeiros registros)
# O método get() traz os dados exatos, sem passar pela inteligência artificial
amostra = colecao_vendas.get(limit=3)

print("\n--- AMOSTRA DOS DADOS GRAVADOS ---")

# Fazendo um loop para imprimir de forma organizada no terminal
for i in range(len(amostra['ids'])):
    print(f"\n🔹 ID do Registro: {amostra['ids'][i]}")
    print(f"🔸 Metadados: {amostra['metadatas'][i]}")
    print(f"📝 Texto do Contexto:\n{amostra['documents'][i]}")
    print("-" * 50)