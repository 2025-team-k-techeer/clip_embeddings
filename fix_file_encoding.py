import os
import unicodedata


def normalize_filename(filename: str) -> str:
    """
    파일명을 유니코드 정규화 (NFC 방식: 한글 조합형 → 완성형)
    """
    return unicodedata.normalize("NFC", filename)


def fix_file_names(folder_path: str):
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        # 디렉토리는 무시
        if not os.path.isfile(old_path):
            continue

        # 유니코드 정규화
        fixed_name = normalize_filename(filename)
        new_path = os.path.join(folder_path, fixed_name)

        try:
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"✅ {filename} → {fixed_name}")
            else:
                print(f"☑️ {filename} (변경 없음)")
        except Exception as e:
            print(f"❌ {filename} 이름 변경 실패: {e}")


if __name__ == "__main__":
    for folder in [
        "real_furniture_image_set/bed",
        "real_furniture_image_set/desk",
        "real_furniture_image_set/couch",
        "real_furniture_image_set/chair",
        "real_furniture_image_set/wardrobe",
        "real_furniture_image_set/hanger",
        "real_furniture_image_set/drawer",
        "real_furniture_image_set/monitor",
        # 필요한 폴더 추가
    ]:
        print(f"\n🧩 폴더 인코딩 복구 중: {folder}")
        fix_file_names(folder)
