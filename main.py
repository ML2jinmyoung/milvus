import os
import glob
import json
from typing import List
import re
from dataclasses import dataclass
from pymilvus import MilvusClient, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import CollectionSchema, FieldSchema
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

CHUNKED_TXT_DIR = "./result"
EMBEDDING_DIM = 1024
DB_NAME = "./kc.db"
COLLECTION_NAME = "kc"
DESCRIPTION = "kc"
DOCUMENT_GROUP = 'law'


@dataclass
class ChunkDocument:
    content: str
    metadata: dict
    chunk_id: int

class ChunkedTextParser:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.embedding_model = BGEM3EmbeddingFunction(
            model_name="BAAI/bge-m3",
            device=device,
            return_dense=True
        )

    def parse_chunked_file(self, file_path: str) -> List[ChunkDocument]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        file_match = re.search(r"<File>(.*?)</File>", text, re.DOTALL)
        path_match = re.search(r"<Path>(.*?)</Path>", text, re.DOTALL)

        if not file_match or not path_match:
            print(f"❌ 메타데이터 누락: {file_path}")
            return []

        file_name = file_match.group(1).strip()
        file_path_value = path_match.group(1).strip()

        metadata = {
            "file": file_name,
            "path": file_path_value,
            "document_group": DOCUMENT_GROUP
        }
        chunks = re.findall(r"<Chunk>(.*?)</Chunk>", text, re.DOTALL)

        parsed_chunks = []
        for idx, chunk_text in enumerate(chunks):
            parsed_chunks.append(ChunkDocument(
                content=chunk_text.strip(),
                metadata=metadata,
                chunk_id=idx
            ))

        return parsed_chunks

    def parse_all(self) -> List[ChunkDocument]:
        all_docs = []
        for txt_file in glob.glob(os.path.join(self.input_dir, "**/*.txt"), recursive=True):
            all_docs.extend(self.parse_chunked_file(txt_file))
        return all_docs



class MilvusLiteInserter:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

        if not self.client.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2048),
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
                params={"nlist": 128}
            )

            self.client.create_index(
                collection_name=collection_name,
                index_params=index_params
            )

    def insert(self, docs: List[ChunkDocument], embedding_model):
        if not docs:
            print("⚠️ No documents to insert")
            return

        contents = [doc.content for doc in docs]
        embeddings_raw = embedding_model.encode_documents(contents)
        embeddings = [vec.tolist() for vec in embeddings_raw["dense"]]

        data = [{
            "metadata": json.dumps(doc.metadata),
            "chunk_id": doc.chunk_id,
            "text": doc.content,
            "vector": emb
        } for doc, emb in zip(docs, embeddings)]

        self.client.insert(self.collection_name, data=data)
        print(f"✅ {len(data)} chunks inserted into '{self.collection_name}'")



def main():
    parser = ChunkedTextParser(CHUNKED_TXT_DIR)
    docs = parser.parse_all()

    inserter = MilvusLiteInserter(COLLECTION_NAME)
    inserter.insert(docs, parser.embedding_model)

if __name__ == "__main__":
    main()
