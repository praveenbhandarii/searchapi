from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from langserve import add_routes
from langchain.vectorstores import Pinecone
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder

app = Flask(__name__)
CORS(app)

index_name = "lawyantra"
openai_api_key = "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"


embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

pinecone.init(
    api_key="a242896b-4f43-484a-9d48-a43fa5a71481",  # find at app.pinecone.io
    environment="us-west4-gcp",  # next to api key in console
)
index = pinecone.Index(index_name)
bm25 = BM25Encoder().default()

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=bm25, index=index, alpha=0.5, top_k=10
)

# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()  # Use request.get_json() to parse JSON data
    query = data.get('query')
    response = retriever.query(query)  # Adjust accordingly
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="localhost", port=8080)
