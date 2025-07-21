import json

INPUT_FILE = "clip_embeddings_json/furniture_embeddings.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

lengths = [len(item["vector"]) for item in data if "vector" in item]

# 벡터 길이 종류 출력
unique_lengths = set(lengths)
print("✅ Unique vector lengths:", unique_lengths)

# 이상한 길이를 가진 항목이 있다면 샘플 출력
if len(unique_lengths) > 1:
    print("\n⚠️ 서로 다른 벡터 길이가 존재합니다! 예시:")
    for item in data:
        l = len(item["vector"])
        if l != list(unique_lengths)[0]:
            print(f"{item['id']}: length = {l}")
else:
    print(f"👍 모든 벡터의 길이는 {list(unique_lengths)[0]}입니다.")
