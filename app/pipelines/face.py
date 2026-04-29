import numpy as np
from app.models.engine import engine
from app.utils.image import decode_image, assess_quality


def extract_embedding(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """Detect face, extract 512D embedding. Raises ValueError if 0 or >1 faces."""
    faces = engine.face_app.get(img)

    if len(faces) == 0:
        raise ValueError("FACE_NOT_DETECTED")
    if len(faces) > 1:
        raise ValueError("MULTIPLE_FACES_DETECTED")

    embedding = faces[0].normed_embedding
    quality = assess_quality(img)
    return embedding, quality


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def apply_threshold(score: float, pass_threshold: float, review_threshold: float) -> str:
    if score >= pass_threshold:
        return "match"
    elif score >= review_threshold:
        return "review"
    return "no_match"


def run_face_match(image_a_bytes: bytes, image_b_bytes: bytes, config) -> dict:
    img_a = decode_image(image_a_bytes)
    img_b = decode_image(image_b_bytes)

    embedding_a, quality_a = extract_embedding(img_a)
    embedding_b, quality_b = extract_embedding(img_b)

    score = cosine_similarity(embedding_a, embedding_b)
    decision = apply_threshold(score, config.FACE_MATCH_PASS_THRESHOLD, config.FACE_MATCH_REVIEW_THRESHOLD)

    del img_a, img_b, embedding_a, embedding_b

    return {
        "match_score": round(score, 4),
        "decision": decision,
        "face_detected_a": True,
        "face_detected_b": True,
        "quality_flags": {
            "image_a": quality_a,
            "image_b": quality_b,
        },
    }
