import insightface
from insightface.app import FaceAnalysis
from paddleocr import PaddleOCR

from app.utils.logger import logger


class ModelEngine:
    _instance = None

    def __init__(self):
        self.face_app = None
        self.ocr_reader = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        logger.info({"event": "model_loading_started"})

        # Load InsightFace
        logger.info({"event": "loading_insightface"})
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info({"event": "insightface_loaded"})

        # Load EasyOCR
        logger.info({"event": "loading_easyocr"})
        self.ocr_reader = PaddleOCR(lang="en")
        logger.info({"event": "easyocr_loaded"})

        logger.info({"event": "all_models_loaded"})

    @property
    def is_ready(self) -> bool:
        return self.face_app is not None and self.ocr_reader is not None


# Singleton instance
engine = ModelEngine.get_instance()