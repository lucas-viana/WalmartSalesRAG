import os
import pandas as pd
import google.generativeai as genai
from flask import Flask, jsonify
from dotenv import load_dotenv

# 1. Carregando as variáveis de ambiente (Protegendo a API Key)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 2. Inicializando o Servidor Flask
app = Flask(__name__)

# 3. Rota de Teste para gerar um Embedding
@app.route('/testar-embedding', methods=['GET'])
def testar_embedding():
    try:
        # Lendo apenas a primeira linha do CSV que criamos no passo anterior
        df = pd.read_csv('dataset_walmart_rag_ptbr.csv')
        texto_exemplo = df['texto_para_embedding'].iloc[0]
        
        # Chamando a API do Google para gerar o Embedding
        # O modelo 'text-embedding-004' é o mais moderno e barato do Google para essa tarefa
        resultado = genai.embed_content(
            model="models/gemini-embedding-2",
            content=texto_exemplo,
            task_type="retrieval_document" # Diz ao Google que estamos guardando isso para busca
        )
        
        vetor = resultado['embedding']
        
        return jsonify({
            "status": "sucesso",
            "texto_original": texto_exemplo,
            "tamanho_do_vetor": len(vetor), # Geralmente retorna 768 dimensões numéricas
            "amostra_dos_numeros": vetor[:5] # Mostrando só os 5 primeiros pra não travar a tela
        })
        
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == '__main__':
    # Rodando o servidor localmente
    app.run(debug=True, port=5000)