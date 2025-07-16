import pandas as pd
from datetime import datetime

# CSV 파일에서 이미지 URL을 분할하여 각 이미지별로 행을 확장하는 스크립트입니다.
# 가구 타입: wardrobe // 원하는 걸로 변경 가능

# CSV 로드
df = pd.read_csv("furnitures_csv\danawa_wardrobe_2025-07-14.csv")

# 이미지별로 분할 저장할 리스트
expanded_rows = []

# 반복
for _, row in df.iterrows():
    image_urls = str(row["image_url"]).split(";")
    image_paths = str(row["image_path"]).split(";")

    # 이미지 수만큼 반복
    for i in range(len(image_urls)):
        if not image_urls[i].strip():  # 빈 문자열 제외
            continue

        expanded_rows.append(
            {
                "name": row["name"],
                "product_url": row["product_url"],
                "image_url": image_urls[i].strip(),
                "image_path": image_paths[i].strip() if i < len(image_paths) else "",
                "width": row.get("width", ""),
                "depth": row.get("depth", ""),
                "height": row.get("height", ""),
                "category": row["category"],
            }
        )

# 결과 DataFrame
image_df = pd.DataFrame(expanded_rows)

# 확인
print(image_df.head())
print(f"전체 이미지 수: {len(image_df)}")

# 오늘 날짜 붙이기 (형식: YYYY-MM-DD)
today = datetime.today().strftime("%m_%d")

# 저장(Optional)
image_df.to_csv(f"real_furnitures_csv/wardrobe_{today}.csv", index=False)
