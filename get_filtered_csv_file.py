import os
import pandas as pd
import unicodedata


# 이미지 파일의 유니코드 정규화 및 공백 제거 후, CSV 파일에서 해당 이미지가 존재하는지 확인하고 필터링하는 스크립트입니다.
def run_filtered(furniture_type, csv_file, image_dir):
    # 가구 타입 및 파일 경로
    IMAGE_DIR = image_dir

    # ✅ 유니코드 정규화 및 공백 제거 함수
    def normalize(s):
        return unicodedata.normalize("NFKC", s).strip()

    # ✅ 실제 존재하는 이미지 파일명 (확장자 제거 + 정규화)
    existing_filenames = {
        normalize(os.path.splitext(f)[0])
        for f in os.listdir(IMAGE_DIR)
        if os.path.isfile(os.path.join(IMAGE_DIR, f))
    }

    # ✅ CSV 로드
    df = pd.read_csv(csv_file)

    # ✅ 비교용 파일명 생성 및 존재 여부 판단
    def filename_exists(image_path):
        if pd.isna(image_path):  # NaN이면 False 반환
            return False
        filename = normalize(os.path.splitext(os.path.basename(image_path))[0])
        return filename in existing_filenames

    # ✅ 존재 여부 적용
    df["file_exists"] = df["image_path"].apply(filename_exists)

    # ✅ 존재하는 행만 필터링
    filtered_df = df[df["file_exists"]].drop(columns=["file_exists"])

    # ✅ 저장 경로 구성
    output_dir = "filtered_furnitures_csv"
    os.makedirs(output_dir, exist_ok=True)
    filename_only = os.path.basename(csv_file)
    save_path = os.path.join(output_dir, f"filtered_{filename_only}")

    # ✅ 저장
    filtered_df.to_csv(save_path, index=False)

    print(
        f"{furniture_type} ✅ 필터링 완료: {len(df)}개 → {len(filtered_df)}개 (인코딩/정규화 처리 포함)"
    )
    return len(filtered_df)


sum = 0
# 가구 타입별 CSV 파일 및 이미지 디렉토리 설정
for furniture_type in [
    "bed",
    "chair",
    "couch",
    "desk",
    "drawer",
    "hanger",
    "monitor",
    "wardrobe",
]:
    csv_file = f"real_furnitures_csv/{furniture_type}_07_17.csv"
    IMAGE_DIR = f"real_furniture_image_set/{furniture_type}"
    sum += run_filtered(furniture_type, csv_file, IMAGE_DIR)
print(f"총 필터링된 이미지 수: {sum}")
