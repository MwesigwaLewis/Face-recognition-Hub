from .face_engine import FaceEngine
from . import database


def recognize(frame, engine=None, threshold=0.65):
    engine = engine or FaceEngine()
    if engine.model is None:
        engine.load_model()
    people = database.get_people()
    results = []

    for face in engine.get_faces(frame):
        best_match = "Unknown"
        highest_score = 0.0
        for name, saved_embeddings in people:
            for saved_embedding in saved_embeddings:
                score = engine.compare_faces(face.embedding, saved_embedding)
                highest_score = max(highest_score, score)
                if score >= highest_score:
                    best_match = name
        if highest_score < threshold:
            best_match = "Unknown"
        results.append({
            "name": best_match,
            "score": highest_score,
            "bbox": face.bbox,
            "landmarks": face.kps,
            "face": face,
        })
    return results
