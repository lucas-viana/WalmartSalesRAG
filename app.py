import os
from flask import Flask, request, jsonify
from flask_cors import CORS # Permite que o React acesse a API
import chromadb
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

# 1. Configurações Iniciais
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
CORS(app) # Liberando acesso para o Frontend

# 2. Nossa Função Customizada (A mesma que usamos para gravar)
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

funcao_embedding = CustomGoogleEmbeddingFunction(api_key=GOOGLE_API_KEY)

# 3. Conectando ao Banco de Dados Existente
chroma_client = chromadb.PersistentClient(path="./chroma_data")
colecao_vendas = chroma_client.get_collection(
    name="walmart_vendas",
    embedding_function=funcao_embedding
)

# 4. Configurando o Modelo de Texto (A IA que vai falar com o usuário)
modelo_texto = genai.GenerativeModel('gemini-2.5-flash')

# 5. A Rota Principal do Chatbot
@app.route('/perguntar', methods=['POST'])
def fazer_pergunta():
    try:
        dados = request.json
        pergunta_usuario = dados.get('pergunta')
        
        if not pergunta_usuario:
            return jsonify({"erro": "A pergunta é obrigatória."}), 400

        # PASSO A: Buscar no banco vetorial
        # O Chroma transforma a pergunta em números sozinho e acha os 3 textos mais próximos
        resultados_busca = colecao_vendas.query(
            query_texts=[pergunta_usuario],
            n_results=53 
        )
        
        # Extraindo os textos encontrados do resultado complexo do Chroma
        textos_encontrados = resultados_busca['documents'][0]
        contexto_unido = "\n".join(textos_encontrados)
        
        # PASSO B: Engenharia de Prompt
        prompt_final = f"""Você é um analista financeiro sênior de uma gigante do varejo.
        Responda à pergunta do usuário baseando-se ÚNICA E EXCLUSIVAMENTE nas informações de contexto abaixo.
        Se a informação não estiver no contexto, diga que não tem dados suficientes. Não invente valores.
        
        CONTEXTO OBTIDO DO BANCO DE DADOS:
        {contexto_unido}
        
        PERGUNTA DO USUÁRIO: {pergunta_usuario}
        
        Sua resposta (seja analítico, claro e direto):"""
        
        # PASSO C: Gerar a resposta com o Gemini
        resposta_ia = modelo_texto.generate_content(prompt_final)
        
        return jsonify({
            "status": "sucesso",
            "resposta": resposta_ia.text,
            "fontes_utilizadas": textos_encontrados # Enviamos as fontes para mostrar no front-end depois!
        })
        
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == '__main__':
    print("Servidor RAG no ar! Aguardando perguntas...")
    app.run(debug=True, port=5000)