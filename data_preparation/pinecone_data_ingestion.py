import json
import os
import boto3
import torch
from pinecone import Pinecone
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection
from transformers import AutoTokenizer, AutoModel

# --- AWS CONFIGURATION ---
# These environment variables will be used by boto3 to find your credentials
AWS_PROFILE = os.getenv("AWS_PROFILE_NAME", "sigir-participant")
AWS_REGION = os.getenv("AWS_REGION_NAME", "us-east-1")

def get_ssm_value(parameter_name, decryption=False):
    """Retrieves a parameter value from AWS SSM Parameter Store."""
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    ssm = session.client("ssm")
    response = ssm.get_parameter(Name=parameter_name, WithDecryption=decryption)
    return response["Parameter"]["Value"]

# --- INGESTION CLASS ---
class DataMorganaIngestor:
    def __init__(self):
        print("--- Initializing Ingestor ---")
        
        # 1. Load Embedding Model (E5-base-v2)
        # E5 is highly effective for RAG but requires specific prefixes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "intfloat/e5-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        print(f"Model {self.model_name} loaded on {self.device}")

        # 2. Connect to Pinecone
        pinecone_key = get_ssm_value("/pinecone/ro_token", decryption=True)
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "ragenta-index")
        self.pc = Pinecone(api_key=pinecone_key)
        self.pinecone_index = self.pc.Index(self.pinecone_index_name)
        print(f"Connected to Pinecone (Index: {self.pinecone_index_name})")

        # 3. Connect to OpenSearch
        # We use IAM-based authentication (AWSV4SignerAuth)
        os_host = get_ssm_value("/opensearch/endpoint")
        credentials = boto3.Session(profile_name=AWS_PROFILE).get_credentials()
        auth = AWSV4SignerAuth(credentials, region=AWS_REGION)
        
        self.os_client = OpenSearch(
            hosts=[{"host": os_host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
        print(f"Connected to OpenSearch (Host: {os_host})")

    def _get_embedding(self, text):
        """Generates embeddings with 'passage: ' prefix (required by E5 for indexing)."""
        input_text = f"passage: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, 
                                truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Perform average pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize embeddings for cosine similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings[0].cpu().tolist()

    def run(self, file_path):
        print(f"Starting processing: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                
                # Extract and clean context
                context_list = data.get('context', [])
                full_text = " ".join(context_list) if isinstance(context_list, list) else context_list
                doc_id = f"morgana_{i}"

                # A. Vector Embedding & Upload to Pinecone (Dense Retrieval)
                vector = self._get_embedding(full_text)
                self.pinecone_index.upsert(vectors=[{
                    "id": doc_id,
                    "values": vector,
                    "metadata": {"text": full_text, "source": "DataMorgana"}
                }])

                # B. Text Indexing to OpenSearch (Sparse Retrieval)
                self.os_client.index(
                    index=self.pinecone_index_name,
                    body={"text": full_text, "doc_id": doc_id},
                    refresh=True
                )

                if i % 5 == 0:
                    print(f"Documents processed: {i+1}")

        print("--- Ingestion completed successfully! ---")

if __name__ == "__main__":
    # Ensure the .jsonl file is in the same directory
    FILE_NAME = "data_preparation/Generation_from_DataMorgana.jsonl"
    if os.path.exists(FILE_NAME):
        ingestor = DataMorganaIngestor()
        ingestor.run(FILE_NAME)
    else:
        print(f"Error: File {FILE_NAME} not found.")