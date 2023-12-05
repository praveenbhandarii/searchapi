"""Example LangChain server exposes a retriever."""
from fastapi import FastAPI
from langchain.embeddings import OpenAIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
import pinecone
from langserve import add_routes
from langchain.vectorstores import Pinecone
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder

index_name = "lawyantra"
openai_api_key="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"

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# initialize pinecone
pinecone.init(
    api_key="a242896b-4f43-484a-9d48-a43fa5a71481",  # find at app.pinecone.io
    # api_key='a242896b-4f43-484a-9d48-a43fa5a71481',
    environment="us-west4-gcp",  # next to api key in console
)
index = pinecone.Index(index_name)
bm25= BM25Encoder().default()

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=bm25, index=index,alpha=0.5,top_k=10
)
# Create store from existing index
#vectorstore = Pinecone.from_existing_index(index_name, embeddings, "context")

#retriever = vectorstore.as_retriever()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:3001","http://your-react-app-origin.com"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

add_routes(app, retriever,path="/chat")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080)