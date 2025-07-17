from dotenv import load_dotenv
import os
import pandas as pd
from google.cloud import storage


# ✅ 1. GCS에 이미지 업로드하고 public URL 생성
def upload_images_to_gcs(bucket_name: str, local_dir: str, gcs_prefix: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    uploaded_urls = {}

    for filename in os.listdir(local_dir):
        local_path = os.path.join(local_dir, filename)
        if not os.path.isfile(local_path):
            continue

        blob_path = f"{gcs_prefix}/{filename}"
        blob = bucket.blob(blob_path)

        # 중복 방지: 이미 업로드된 파일은 건너뜜
        if not blob.exists():
            blob.upload_from_filename(local_path)

        # ✅ 수정: blob_path를 인코딩하지 않고 직접 사용하여 URL 생성
        # GCS public URL은 object_name 자체를 URL 인코딩하지 않습니다 (슬래시 포함).
        raw_url = f"https://storage.googleapis.com/{bucket_name}/{blob_path}"
        base_name = os.path.splitext(filename)[0]
        uploaded_urls[base_name] = raw_url

    return uploaded_urls


# ✅ 2. 기존 CSV의 image_path를 기반으로 GCS URL 매핑 후 업데이트
def update_image_urls_with_gcs(
    csv_input_path: str,
    uploaded_urls: dict,
    output_csv_path: str,
):
    if not os.path.exists(csv_input_path):
        print(f"❌ CSV 파일 없음: {csv_input_path}")
        return

    df = pd.read_csv(csv_input_path)

    def get_gcs_url(image_path):
        if pd.isna(image_path):
            return ""
        base_name = os.path.splitext(os.path.basename(str(image_path)))[0]
        return uploaded_urls.get(base_name, "")

    df["image_url"] = df["image_path"].apply(get_gcs_url)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ CSV 저장 완료: {output_csv_path}")


# ✅ 3. 실행 부분
if __name__ == "__main__":
    load_dotenv()

    # 환경 변수에서 인증 파일 경로 가져오기
    credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credential_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
    else:
        raise EnvironmentError(
            "❌ GOOGLE_APPLICATION_CREDENTIALS 환경변수가 설정되지 않았습니다."
        )

    interior_types = [
        # "bed",
        # "chair",
        "couch",
        "desk",
        "drawer",
        "wardrobe",
        "hanger",
        "monitor",
    ]

    gcs_bucket_name = "furniture-image-bucket"

    for interior_type in interior_types:
        image_dir = f"real_furniture_image_set/{interior_type}"
        csv_input = f"real_furnitures_csv/filtered_{interior_type}_07_17.csv"
        csv_output = f"gcs_furniture_csv/{interior_type}_gcs.csv"
        gcs_folder = interior_type

        print(f"\n🚀 {interior_type.upper()} 처리 시작")

        # ✅ 1. GCS에 이미지 업로드
        uploaded_urls = upload_images_to_gcs(
            bucket_name=gcs_bucket_name, local_dir=image_dir, gcs_prefix=gcs_folder
        )

        # ✅ 2. image_url 열 업데이트 후 CSV 저장
        update_image_urls_with_gcs(
            csv_input_path=csv_input,
            uploaded_urls=uploaded_urls,
            output_csv_path=csv_output,
        )
