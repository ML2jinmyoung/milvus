from pymilvus import MilvusClient, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import pprint
import torch
import json

COLLECTION_NAME = "kc"
QUESTIONS = ["상장회사 자기주식 취득을 위한 절차는?", "계열사간 이사, 감사 겸직시 유의사항은?", "상근감사 연임은 언제까지 가능한가?",   "하도급 계약시 서면 미교부는 어떤 문제가 발생하는가?", "사내하도급시 유의할 사항은?"]

# device = "mps" if torch.backends.mps.is_available() else "cpu"

def search_documents(query: str, top_k: int = 3):
    client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
    
    client.load_collection(COLLECTION_NAME)
    
    embedding_model = BGEM3EmbeddingFunction(
        model_name="BAAI/bge-m3",
        device="cuda:0",
        # device=device,
        return_dense=True
    )
    
    query_vector = embedding_model.encode_documents([query])["dense"][0].tolist()
    
    search_params = {
        "metric_type": "COSINE",
        "offset": 0,
        "ignore_growing": False,
        "params": {"nprobe": 10}
    }
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        output_fields=["text", "metadata"],
        limit=top_k,
        **search_params
    )

    print(f"\n🔍 검색 쿼리: '{query}'\n")
    
    for i, hits in enumerate(results):
        print(f"\n=== 검색 결과 ===")

        for hit in hits:
            entity = hit["entity"]  
            print(f"📂 정보: { json.loads(entity["metadata"])}")
            print(f"유사도 점수: {hit['distance']:.4f}")
            print("\n📝 내용:")
            print(entity["text"])
            print("\n" + "="*50)
            # pprint.pprint(hit)

if __name__ == "__main__":
    query = QUESTIONS[2]
    search_documents(query, top_k=3)
