# import os
# import json
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import onnxruntime as ort

# # 경로 설정
# PRODUCTS_JSON = "danawa_products.json"
# IMG_DIR = "real_furniture_image_set"
# OUTPUT_DIR = "clip_embeddings_json"
# ALL_EMBEDDINGS_FILE = "furniture_embeddings.json"
# MISSING_FILE = "missing_images.json"
# MISSING_FIELDS_FILE = "missing_fields_records.json"
# BATCH_SIZE = 32
# MODEL_PATH = "models/image_clip.onnx"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ONNX 모델 로드
# session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
# input_name = session.get_inputs()[0].name

# missing_images = []
# missing_fields_records = []
# all_embeddings = []


# # 전처리 함수 (배치 차원 없이)
# def preprocess_image(path):
#     img = Image.open(path).convert("RGB").resize((224, 224))
#     img_np = np.array(img).astype(np.float32) / 255.0
#     img_np = (img_np - 0.5) / 0.5
#     img_np = np.transpose(img_np, (2, 0, 1))
#     return img_np


# def get_filename_from_url(url):
#     if not isinstance(url, str):
#         url = str(url)
#     return os.path.basename(url)


# def extract_metadata(prod, image_url, filename):
#     def parse_json_field(val):
#         if isinstance(val, str):
#             try:
#                 return json.loads(val)
#             except Exception:
#                 return None
#         return val if val else None

#     return {
#         "_id": prod.get("_id"),
#         "label": prod.get("label"),
#         "product_name": prod.get("product_name"),
#         "product_url": prod.get("product_url"),
#         "image_url": image_url,
#         "dimensions": parse_json_field(prod.get("dimensions")),
#         "created_at": prod.get("created_at"),
#         "updated_at": prod.get("updated_at"),
#         "filename": filename,
#     }


# def main():
#     # 1. 제품 데이터 로드
#     with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
#         products = json.load(f)
#     batch_images = []
#     batch_meta = []
#     idx = 0
#     for prod in tqdm(products, desc="제품별 처리"):
#         image_urls = prod.get("image_url", [])
#         if isinstance(image_urls, str):
#             try:
#                 image_urls = json.loads(image_urls)
#             except Exception:
#                 image_urls = [image_urls]
#         for img_url in image_urls:
#             filename = get_filename_from_url(img_url)
#             # label(카테고리) 폴더에서 이미지 탐색
#             category = prod.get("label")
#             local_img_path = os.path.join(IMG_DIR, category, filename)
#             if not (
#                 isinstance(local_img_path, str)
#                 and local_img_path
#                 and os.path.exists(local_img_path)
#             ):
#                 missing_images.append(
#                     {
#                         "category": category,
#                         "image_url": img_url,
#                         "filename": filename,
#                         "reason": "file_not_found",
#                     }
#                 )
#                 continue
#             try:
#                 img_np = preprocess_image(local_img_path)
#                 batch_images.append(img_np)
#                 batch_meta.append((prod, img_url, filename, idx, category))
#             except Exception as e:
#                 missing_images.append(
#                     {
#                         "category": category,
#                         "image_url": img_url,
#                         "filename": filename,
#                         "reason": f"open_error: {str(e)}",
#                     }
#                 )
#                 continue
#             if len(batch_images) == BATCH_SIZE:
#                 run_batch_inference(batch_images, batch_meta)
#                 batch_images, batch_meta = [], []
#             idx += 1
#     if len(batch_images) > 0:
#         run_batch_inference(batch_images, batch_meta)
#     # 누락 이미지/필드 저장
#     with open(MISSING_FILE, "w", encoding="utf-8") as f:
#         json.dump(missing_images, f, ensure_ascii=False, indent=2)
#     if missing_fields_records:
#         with open(MISSING_FIELDS_FILE, "w", encoding="utf-8") as f:
#             json.dump(missing_fields_records, f, ensure_ascii=False, indent=2)
#     # 모든 임베딩+메타데이터를 하나의 json 파일로 저장
#     with open(
#         os.path.join(OUTPUT_DIR, ALL_EMBEDDINGS_FILE), "w", encoding="utf-8"
#     ) as f:
#         json.dump(all_embeddings, f, ensure_ascii=False, indent=2)


# def run_batch_inference(batch_images, batch_meta):
#     batch_np = np.stack(batch_images, axis=0)
#     ort_inputs = {input_name: batch_np}
#     ort_outs = np.asarray(session.run(None, ort_inputs)[0])
#     if len(ort_outs.shape) == 3:
#         ort_outs = ort_outs.mean(axis=1)
#     for i, (prod, img_url, filename, idx, category) in enumerate(batch_meta):
#         embedding = ort_outs[i].tolist() if ort_outs is not None else None

#         dims = prod.get("dimensions", {})
#         if isinstance(dims, str):
#             try:
#                 dims = json.loads(dims)
#             except:
#                 dims = {}

#         point = {
#             "id": prod.get("_id"),
#             "vector": embedding,
#             "payload": {
#                 "product_name": prod.get("product_name"),
#                 "label": prod.get("label"),
#                 "product_url": prod.get("product_url"),
#                 "image_url": img_url,
#                 "filename": filename,
#                 "width_cm": dims.get("width_cm"),
#                 "depth_cm": dims.get("depth_cm"),
#                 "height_cm": dims.get("height_cm"),
#                 "created_at": prod.get("created_at"),
#                 "updated_at": prod.get("updated_at"),
#             },
#         }

#         # 누락 체크
#         required_fields = ["id", "vector"]
#         missing = [f for f in required_fields if point.get(f) is None]
#         if missing:
#             print(f"[누락] {prod.get('_id')}_{filename}: {', '.join(missing)}")
#             missing_fields_records.append(
#                 {
#                     "file_name": f"{prod.get('_id')}_{filename}",
#                     "missing_fields": missing,
#                     "record": point,
#                 }
#             )
#         all_embeddings.append(point)


# if __name__ == "__main__":
#     main()
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
from sklearn.preprocessing import normalize  # ✅ 정규화 추가

# 경로 설정
PRODUCTS_JSON = "danawa_products.json"
IMG_DIR = "real_furniture_image_set"
OUTPUT_DIR = "furniture_embeddings(2)"
ALL_EMBEDDINGS_FILE = "furniture_embeddings.json"
MISSING_FILE = "missing_images.json"
MISSING_FIELDS_FILE = "missing_fields_records.json"
BATCH_SIZE = 32
MODEL_PATH = "models/image_clip.onnx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ONNX 모델 로드
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

missing_images = []
missing_fields_records = []
all_embeddings = []


# 전처리 함수 (배치 차원 없이)
def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np


def get_filename_from_url(url):
    if not isinstance(url, str):
        url = str(url)
    return os.path.basename(url)


def extract_metadata(prod, image_url, filename):
    def parse_json_field(val):
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return None
        return val if val else None

    return {
        "_id": prod.get("_id"),
        "label": prod.get("label"),
        "product_name": prod.get("product_name"),
        "product_url": prod.get("product_url"),
        "image_url": image_url,
        "dimensions": parse_json_field(prod.get("dimensions")),
        "created_at": prod.get("created_at"),
        "updated_at": prod.get("updated_at"),
        "filename": filename,
    }


def main():
    with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
        products = json.load(f)
    batch_images = []
    batch_meta = []
    idx = 0
    for prod in tqdm(products, desc="제품별 처리"):
        image_urls = prod.get("image_url", [])
        if isinstance(image_urls, str):
            try:
                image_urls = json.loads(image_urls)
            except Exception:
                image_urls = [image_urls]
        for img_url in image_urls:
            filename = get_filename_from_url(img_url)
            category = prod.get("label")
            local_img_path = os.path.join(IMG_DIR, category, filename)
            if not (
                isinstance(local_img_path, str)
                and local_img_path
                and os.path.exists(local_img_path)
            ):
                missing_images.append(
                    {
                        "category": category,
                        "image_url": img_url,
                        "filename": filename,
                        "reason": "file_not_found",
                    }
                )
                continue
            try:
                img_np = preprocess_image(local_img_path)
                batch_images.append(img_np)
                batch_meta.append((prod, img_url, filename, idx, category))
            except Exception as e:
                missing_images.append(
                    {
                        "category": category,
                        "image_url": img_url,
                        "filename": filename,
                        "reason": f"open_error: {str(e)}",
                    }
                )
                continue
            if len(batch_images) == BATCH_SIZE:
                run_batch_inference(batch_images, batch_meta)
                batch_images, batch_meta = [], []
            idx += 1
    if len(batch_images) > 0:
        run_batch_inference(batch_images, batch_meta)
    with open(MISSING_FILE, "w", encoding="utf-8") as f:
        json.dump(missing_images, f, ensure_ascii=False, indent=2)
    if missing_fields_records:
        with open(MISSING_FIELDS_FILE, "w", encoding="utf-8") as f:
            json.dump(missing_fields_records, f, ensure_ascii=False, indent=2)
    with open(
        os.path.join(OUTPUT_DIR, ALL_EMBEDDINGS_FILE), "w", encoding="utf-8"
    ) as f:
        json.dump(all_embeddings, f, ensure_ascii=False, indent=2)


def run_batch_inference(batch_images, batch_meta):
    batch_np = np.stack(batch_images, axis=0)
    ort_inputs = {input_name: batch_np}
    ort_outs = np.asarray(session.run(None, ort_inputs)[0])

    if len(ort_outs.shape) == 3:
        ort_outs = ort_outs.mean(axis=1)  # (N, 768)

    # ✅ 정규화 수행
    ort_outs = normalize(ort_outs, norm="l2", axis=1)  # (N, 768)

    for i, (prod, img_url, filename, idx, category) in enumerate(batch_meta):
        embedding = ort_outs[i].tolist()

        dims = prod.get("dimensions", {})
        if isinstance(dims, str):
            try:
                dims = json.loads(dims)
            except:
                dims = {}

        point = {
            "id": prod.get("_id"),
            "vector": embedding,
            "payload": {
                "product_name": prod.get("product_name"),
                "label": prod.get("label"),
                "product_url": prod.get("product_url"),
                "image_url": img_url,
                "filename": filename,
                "width_cm": dims.get("width_cm"),
                "depth_cm": dims.get("depth_cm"),
                "height_cm": dims.get("height_cm"),
                "created_at": prod.get("created_at"),
                "updated_at": prod.get("updated_at"),
            },
        }

        required_fields = ["id", "vector"]
        missing = [f for f in required_fields if point.get(f) is None]
        if missing:
            print(f"[누락] {prod.get('_id')}_{filename}: {', '.join(missing)}")
            missing_fields_records.append(
                {
                    "file_name": f"{prod.get('_id')}_{filename}",
                    "missing_fields": missing,
                    "record": point,
                }
            )
        all_embeddings.append(point)


if __name__ == "__main__":
    main()
