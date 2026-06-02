import os
from flask import Flask, request, jsonify
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "vendas-walmart"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash') 

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Bem-vindo ao servidor de perguntas!"})

@app.route('/perguntar', methods=['POST'])
def perguntar():
    try:
        data = request.get_json()
        pergunta = data.get("pergunta")

        if not pergunta:
            return jsonify({"erro": "A chave 'pergunta' é obrigatória no JSON."}), 400

        embedding_response = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[pergunta],
            parameters={"input_type": "query"} 
        )
        vetor_pergunta = embedding_response[0].values

        busca_pinecone = index.query(
            vector=vetor_pergunta,
            top_k=3,
            include_metadata=True 
        )

        contextos = [match['metadata']['texto'] for match in busca_pinecone['matches']]
        contexto_unido = "\n\n".join(contextos)

        prompt_final = f"""Você é um assistente de análise de dados. 
Use EXCLUSIVAMENTE as informações abaixo para responder à pergunta do usuário.
Se a informação não estiver no contexto, diga que não sabe.

Contexto (Dados do Walmart):
{contexto_unido}

Pergunta: {pergunta}
"""
        
        resposta_llm = model.generate_content(prompt_final)

        return jsonify({"resposta": resposta_llm.text}), 200

    except Exception as e:
        return jsonify({"erro_interno": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)