from pymilvus import MilvusClient, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import pprint
import torch

COLLECTION_NAME = "kc"

device = "mps" if torch.backends.mps.is_available() else "cpu"

def search_documents(query: str, top_k: int = 3):
    # Milvus 클라이언트 초기화
    # client = MilvusClient("./kc.db")
    client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
    
    # 컬렉션 로드 (milvus standalone일때 사용)
    client.load_collection(COLLECTION_NAME)
    
    # 임베딩 모델 초기화
    embedding_model = BGEM3EmbeddingFunction(
        model_name="BAAI/bge-m3",
        # device="cuda:0",
        device=device,
        return_dense=True
    )
    
    # 쿼리 벡터 추출
    query_vector = embedding_model.encode_documents([query])["dense"][0].tolist()
    
    # Milvus에서 검색
    search_params = {
        "metric_type": "L2",
        "offset": 0,
        "ignore_growing": False,
        "params": {"nprobe": 10}
    }
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        output_fields=["text", "file_name", "file_path", "chunk_id"],
        limit=top_k,
        **search_params
    )

    print(f"\n🔍 검색 쿼리: '{query}'\n")
    
    for i, hits in enumerate(results):
        print(f"\n=== 검색 결과 ===")

        for hit in hits:
            entity = hit["entity"]  
            print(f"\n document group: {entity['document_group']}")
            print(f"\n📄 문서: {entity['file_name']}")
            print(f"📂 경로: {entity['file_path']}")
            print(f"청크 ID: {entity['chunk_id']}")
            print(f"유사도 점수: {hit['distance']:.4f}")
            print("\n📝 내용:")
            print(entity["text"])
            print("\n" + "="*50)
            # pprint.pprint(hit)

if __name__ == "__main__":
    query = "애순이 관식이"
    search_documents(query, top_k=3)
