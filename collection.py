import os
import glob
import re
from pathlib import Path
from typing import List
from dataclasses import dataclass
from pymilvus import MilvusClient, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import CollectionSchema, FieldSchema
import pprint
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

CHUNKED_TXT_DIR = "./result_test"
EMBEDDING_DIM = 1024
DB_NAME = "./kc.db"
COLLECTION_NAME = "test"
DESCRIPTION = "kc 준법경영실"


@dataclass
class ChunkDocument:
    content: str
    file_name: str
    file_path: str
    chunk_id: int


class ChunkedTextParser:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.embedding_model = BGEM3EmbeddingFunction(
            model_name="BAAI/bge-m3",
            device=device,
            # device="cuda:0",
            return_dense=True
        )

    def parse_chunked_file(self, file_path: str) -> List[ChunkDocument]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        file_name_match = re.search(r'^File:\s*(.+)$', text, re.MULTILINE)
        file_path_match = re.search(r'^Path:\s*(.+)$', text, re.MULTILINE)
        if not file_name_match or not file_path_match:
            print(f"메타데이터 누락: {file_path}")
            return []

        file_name = file_name_match.group(1).strip()
        file_path_value = file_path_match.group(1).strip()

        chunks = re.split(r'^--- Chunk (\d+) ---\s*', text, flags=re.MULTILINE)
        parsed_chunks = []
        for i in range(1, len(chunks), 2):
            chunk_id = int(chunks[i])
            chunk_text = chunks[i + 1].strip()
            parsed_chunks.append(ChunkDocument(
                content=chunk_text,
                file_name=file_name,
                file_path=file_path_value,
                chunk_id=chunk_id
            ))
        return parsed_chunks

    def parse_all(self) -> List[ChunkDocument]:
        all_docs = []
        for txt_file in glob.glob(os.path.join(self.input_dir, "**/*.txt"), recursive=True):
            all_docs.extend(self.parse_chunked_file(txt_file))
        return all_docs


class MilvusLiteInserter:
    def __init__(self, collection_name: str, db_path: str = DB_NAME):
        self.collection_name = collection_name
        # self.client = MilvusClient(uri=db_path) #milvus lite
        self.client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

        if not self.client.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="chunk_id", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            ]
            schema = CollectionSchema(fields=fields, description=DESCRIPTION)
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema
            )

            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type="COSINE",
                index_type="IVF_FLAT",
                index_name="vector_index",
                params={ "nlist": 128 }
            )

            self.client.create_index(
                collection_name=collection_name,
                index_params=index_params
            )

    def insert(self, docs: List[ChunkDocument], embedding_model):
        contents = [doc.content for doc in docs]
        embeddings_raw = embedding_model.encode_documents(contents)
        # dense 벡터 리스트 추출
        embeddings = [vec.tolist() for vec in embeddings_raw["dense"]]
        data = [{
        "file_name": doc.file_name,
        "file_path": doc.file_path,
        "chunk_id": doc.chunk_id,
        "text": doc.content,
        "vector": emb
        } for doc, emb in zip(docs, embeddings)]
        
        if data:
            self.client.insert(self.collection_name, data=data)
            print(f"✅ {len(data)} chunks inserted into '{self.collection_name}'")
        else:
            print("⚠️ No documents to insert")


def main():
    parser = ChunkedTextParser(CHUNKED_TXT_DIR)
    docs = parser.parse_all()

    inserter = MilvusLiteInserter(COLLECTION_NAME)
    inserter.insert(docs, parser.embedding_model)

if __name__ == "__main__":
    main()
