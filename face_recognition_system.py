import argparse
import json
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, ImageDraw


DATABASE_VERSION = "quality_multi_template_v1"
MODEL_NAME = "InceptionResnetV1_vggface2"
DEFAULT_GLOBAL_THRESHOLD = 0.84
DEFAULT_ALPHA = 0.7
DEFAULT_TOP_K = 3


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def get_default_torch_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class FaceRecognitionSystem:
    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            device = get_default_torch_device()
        self.device = torch.device(device)
        self.detector_device = torch.device("cpu") if self.device.type == "mps" else self.device

        # MTCNN completes face detection, landmark localization, alignment, and normalization.
        alignment_padding = 20
        self.mtcnn = MTCNN(
            160,
            alignment_padding,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=True,
            device=self.detector_device,
        )

        # CNN embedding network pretrained on VGGFace2.
        self.feature_extractor = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # Serialise database write operations to prevent concurrent corruption.
        self._db_lock = threading.Lock()

    def _detect_align_face_with_confidence(
        self, image_path: str
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        image = Image.open(image_path).convert("RGB")
        try:
            boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        except ValueError as exc:
            if "expected a non-empty list of Tensors" not in str(exc):
                raise
            warnings.warn(
                f"Skipping image with no MTCNN candidate boxes: {image_path}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None, None, None, None

        if boxes is None or probs is None or len(boxes) == 0:
            return None, None, None, None

        # Use the largest high-confidence face as the target face.
        best_index = None
        best_area = -1.0
        for idx, (box, score) in enumerate(zip(boxes, probs)):
            if score is None or score < 0.90:
                continue
            x1, y1, x2, y2 = box
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best_index = idx

        if best_index is None:
            best_index = int(np.argmax(probs))

        face_tensor = self.mtcnn.extract(image, [boxes[best_index]], save_path=None)
        if face_tensor is None:
            return None, None, None, None

        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)

        return (
            face_tensor.to(self.device),
            boxes[best_index],
            landmarks[best_index],
            float(probs[best_index]),
        )

    def detect_align_face(
        self, image_path: str
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[np.ndarray]]:
        aligned_face, box, landmarks, _ = self._detect_align_face_with_confidence(image_path)
        return aligned_face, box, landmarks

    @torch.no_grad()
    def extract_feature(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        feature, box, landmarks, _ = self.extract_template(image_path)
        return feature, box, landmarks

    @torch.no_grad()
    def extract_template(
        self, image_path: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        aligned_face, box, landmarks, confidence = self._detect_align_face_with_confidence(image_path)
        if aligned_face is None:
            return None, None, None, None

        embedding = self.feature_extractor(aligned_face)
        feature = embedding.squeeze(0).cpu().numpy().astype(np.float32)
        feature = l2_normalize(feature)
        return feature, box, landmarks, confidence

    @staticmethod
    def save_database(
        output_path: str,
        identity_names: np.ndarray,
        templates: np.ndarray,
        template_labels: np.ndarray,
        template_quality_scores: np.ndarray,
        template_source_images: np.ndarray,
        global_threshold: float = DEFAULT_GLOBAL_THRESHOLD,
    ) -> None:
        """Save the structured face database.

        The NPZ fields are intentionally explicit:
        identity_names lists registered people; templates stores every embedding;
        template_labels maps each embedding to a person; template_quality_scores
        records template reliability; template_source_images preserves traceability;
        thresholds and model/version fields make recognition reproducible.
        """
        embedding_dim = int(templates.shape[1]) if templates.ndim == 2 else 0
        np.savez_compressed(
            output_path,
            identity_names=identity_names.astype(str),
            templates=templates.astype(np.float32),
            template_labels=template_labels.astype(str),
            template_quality_scores=template_quality_scores.astype(np.float32),
            template_source_images=template_source_images.astype(str),
            global_threshold=np.array([global_threshold], dtype=np.float32),
            embedding_dim=np.array([embedding_dim], dtype=np.int32),
            model_name=np.array([MODEL_NAME]),
            database_version=np.array([DATABASE_VERSION]),
        )

    @staticmethod
    def load_database(database_path: str) -> Dict[str, Any]:
        if not Path(database_path).exists():
            raise FileNotFoundError(f"Database file does not exist: {database_path}")

        data = np.load(database_path, allow_pickle=True)
        files = set(data.files)
        if {"identity_names", "templates", "template_labels"}.issubset(files):
            templates = data["templates"].astype(np.float32)
            return {
                "identity_names": data["identity_names"].astype(str),
                "templates": templates,
                "template_labels": data["template_labels"].astype(str),
                "template_quality_scores": data["template_quality_scores"].astype(np.float32),
                "template_source_images": data["template_source_images"].astype(str),
                "global_threshold": float(data["global_threshold"][0]),
                "embedding_dim": int(data["embedding_dim"][0]),
                "model_name": str(data["model_name"][0]),
                "database_version": str(data["database_version"][0]),
            }

        if {"labels", "embeddings"}.issubset(files):
            labels = data["labels"].astype(str)
            embeddings = data["embeddings"].astype(np.float32)
            identity_names = np.unique(labels)
            return {
                "identity_names": identity_names,
                "templates": embeddings,
                "template_labels": labels,
                "template_quality_scores": np.ones(len(labels), dtype=np.float32),
                "template_source_images": np.array(["unknown"] * len(labels)),
                "global_threshold": DEFAULT_GLOBAL_THRESHOLD,
                "embedding_dim": int(embeddings.shape[1]),
                "model_name": MODEL_NAME,
                "database_version": "legacy_labels_embeddings",
            }

        raise ValueError(f"Unsupported database format: {database_path}")

    def build_database(
        self,
        dataset_dir: str,
        output_path: str,
        global_threshold: float = DEFAULT_GLOBAL_THRESHOLD,
    ) -> Dict[str, np.ndarray]:
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_dir}")

        templates: Dict[str, np.ndarray] = {}
        quality_scores: List[float] = []
        source_images: List[str] = []
        stats: Dict[str, int] = {}

        for person_dir in sorted(dataset_path.iterdir()):
            if not person_dir.is_dir():
                continue

            person_features: List[np.ndarray] = []
            for image_path in sorted(person_dir.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                feature, _, _, confidence = self.extract_template(str(image_path))
                if feature is not None:
                    person_features.append(feature)
                    quality_scores.append(float(confidence) if confidence is not None else 1.0)
                    source_images.append(str(image_path.resolve()))

            if not person_features:
                continue

            templates[person_dir.name] = np.vstack(person_features).astype(np.float32)
            stats[person_dir.name] = len(person_features)

        if not templates:
            raise RuntimeError("No valid faces were extracted from the dataset.")

        labels = np.array(
            [name for name, person_templates in templates.items() for _ in range(len(person_templates))]
        )
        template_array = np.vstack(list(templates.values())).astype(np.float32)
        identity_names = np.array(list(templates.keys()))
        self.save_database(
            output_path=output_path,
            identity_names=identity_names,
            templates=template_array,
            template_labels=labels,
            template_quality_scores=np.array(quality_scores, dtype=np.float32),
            template_source_images=np.array(source_images),
            global_threshold=global_threshold,
        )

        meta_path = Path(output_path).with_suffix(".json")
        meta = {
            "dataset_dir": str(dataset_path.resolve()),
            "num_identities": len(templates),
            "num_templates": int(len(labels)),
            "images_per_identity": stats,
            "feature_dim": int(template_array.shape[1]),
            "model": MODEL_NAME,
            "detector": "MTCNN",
            "similarity": "weighted_multi_template_cosine",
            "database_version": DATABASE_VERSION,
            "global_threshold": global_threshold,
            "matching": {
                "method": "weighted_multi_template",
                "top1_weight": DEFAULT_ALPHA,
                "topk_mean_weight": 1.0 - DEFAULT_ALPHA,
                "topk": DEFAULT_TOP_K,
            },
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return templates

    # ------------------------------------------------------------------
    # Incremental database operations
    # ------------------------------------------------------------------

    def _extract_templates_from_dir(self, person_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract templates, quality scores, and source paths from a directory."""
        person_templates: List[np.ndarray] = []
        quality_scores: List[float] = []
        source_images: List[str] = []
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            feature, _, _, confidence = self.extract_template(str(image_path))
            if feature is not None:
                person_templates.append(feature)
                quality_scores.append(float(confidence) if confidence is not None else 1.0)
                source_images.append(str(image_path.resolve()))
        if not person_templates:
            return (
                np.empty((0, 512), dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=str),
            )
        return (
            np.vstack(person_templates).astype(np.float32),
            np.array(quality_scores, dtype=np.float32),
            np.array(source_images),
        )

    def _load_or_create_meta(self, db_path: Path) -> dict:
        """Return existing metadata dict, or a sensible default for a new database."""
        meta_path = db_path.with_suffix(".json")
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "dataset_dir": "",
            "num_identities": 0,
            "num_templates": 0,
            "images_per_identity": {},
            "feature_dim": 512,
            "model": MODEL_NAME,
            "detector": "MTCNN",
            "similarity": "weighted_multi_template_cosine",
            "database_version": DATABASE_VERSION,
            "global_threshold": DEFAULT_GLOBAL_THRESHOLD,
            "matching": {
                "method": "weighted_multi_template",
                "top1_weight": DEFAULT_ALPHA,
                "topk_mean_weight": 1.0 - DEFAULT_ALPHA,
                "topk": DEFAULT_TOP_K,
            },
        }

    def _write_metadata(self, db_path: Path, database: Dict[str, Any], meta: dict) -> None:
        meta.update(
            {
                "num_identities": int(len(database["identity_names"])),
                "num_templates": int(len(database["templates"])),
                "feature_dim": int(database["embedding_dim"]),
                "model": database["model_name"],
                "detector": "MTCNN",
                "similarity": "weighted_multi_template_cosine",
                "database_version": database["database_version"],
                "global_threshold": float(database["global_threshold"]),
                "matching": {
                    "method": "weighted_multi_template",
                    "top1_weight": DEFAULT_ALPHA,
                    "topk_mean_weight": 1.0 - DEFAULT_ALPHA,
                    "topk": DEFAULT_TOP_K,
                },
            }
        )
        db_path.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def add_identity(
        self, name: str, person_dir: str, database_path: str,
    ) -> dict:
        """Extract features for a **new** identity and append its templates to the
        database.

        ``person_dir`` must be a directory containing one or more face images
        (``.jpg`` / ``.png`` / ``.bmp``).  If the database file does not exist
        yet it is created from scratch.

        Returns a summary dict with keys ``name``, ``num_images``,
        ``num_identities``, and ``num_templates``.

        Raises ``ValueError`` when *name* already exists in the database.
        """
        person_path = Path(person_dir)
        if not person_path.is_dir():
            raise FileNotFoundError(f"Person directory not found: {person_dir}")

        db_path = Path(database_path)

        with self._db_lock:
            person_templates, person_quality, person_sources = self._extract_templates_from_dir(person_path)
            if len(person_templates) == 0:
                raise RuntimeError(f"No valid faces found in {person_dir}")

            meta = self._load_or_create_meta(db_path)
            if db_path.exists():
                database = self.load_database(str(db_path))
                if name in database["identity_names"]:
                    raise ValueError(
                        f"Identity '{name}' already exists in the database. "
                        f"Use update_identity() to add more images."
                    )
                identity_names = np.append(database["identity_names"], name)
                all_templates = np.vstack([database["templates"], person_templates])
                all_labels = np.append(database["template_labels"], np.array([name] * len(person_templates)))
                all_quality = np.append(database["template_quality_scores"], person_quality)
                all_sources = np.append(database["template_source_images"], person_sources)
                meta["dataset_dir"] = meta.get("dataset_dir") or str(person_path.parent.resolve())
                global_threshold = float(database["global_threshold"])
            else:
                identity_names = np.array([name])
                all_templates = person_templates
                all_labels = np.array([name] * len(person_templates))
                all_quality = person_quality
                all_sources = person_sources
                meta["dataset_dir"] = str(person_path.parent.resolve())
                global_threshold = DEFAULT_GLOBAL_THRESHOLD

            self.save_database(
                str(db_path), identity_names, all_templates, all_labels,
                all_quality, all_sources, global_threshold,
            )
            database = self.load_database(str(db_path))
            meta.setdefault("images_per_identity", {})[name] = len(person_templates)
            self._write_metadata(db_path, database, meta)

        return {
            "name": name,
            "num_images": len(person_templates),
            "num_identities": len(identity_names),
            "num_templates": len(all_labels),
        }

    def update_identity(
        self, name: str, person_dir: str, database_path: str,
    ) -> dict:
        """Re-compute the templates for an **existing** identity using all images
        currently present in ``person_dir`` (including any newly added ones).

        The database must already exist and contain *name*.

        Returns a summary dict with keys ``name``, ``num_images``,
        ``num_identities``, and ``num_templates``.

        Raises ``ValueError`` when *name* is not found in the database.
        """
        person_path = Path(person_dir)
        if not person_path.is_dir():
            raise FileNotFoundError(f"Person directory not found: {person_dir}")

        db_path = Path(database_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {database_path}")

        with self._db_lock:
            database = self.load_database(str(db_path))
            labels = database["template_labels"]
            indices = np.where(labels == name)[0]
            if len(indices) == 0:
                raise ValueError(
                    f"Identity '{name}' not found in the database. "
                    f"Use add_identity() to register a new identity."
                )
            person_templates, person_quality, person_sources = self._extract_templates_from_dir(person_path)
            if len(person_templates) == 0:
                raise RuntimeError(f"No valid faces found in {person_dir}")

            keep_mask = labels != name
            labels = np.append(labels[keep_mask], np.array([name] * len(person_templates)))
            templates = np.vstack([database["templates"][keep_mask], person_templates])
            quality = np.append(database["template_quality_scores"][keep_mask], person_quality)
            sources = np.append(database["template_source_images"][keep_mask], person_sources)

            meta = self._load_or_create_meta(db_path)
            self.save_database(
                str(db_path),
                database["identity_names"],
                templates,
                labels,
                quality,
                sources,
                float(database["global_threshold"]),
            )
            database = self.load_database(str(db_path))
            meta.setdefault("images_per_identity", {})[name] = len(person_templates)
            self._write_metadata(db_path, database, meta)

        return {
            "name": name,
            "num_images": len(person_templates),
            "num_identities": len(database["identity_names"]),
            "num_templates": len(labels),
        }

    def remove_identity(self, name: str, database_path: str) -> dict:
        """Remove an identity and all of its templates from the database.

        Returns a summary dict with keys ``name``, ``num_identities``, and
        ``num_templates``.

        Raises ``ValueError`` when *name* is not found in the database.
        """
        db_path = Path(database_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {database_path}")

        with self._db_lock:
            database = self.load_database(str(db_path))
            labels = database["template_labels"]
            indices = np.where(labels == name)[0]
            if len(indices) == 0:
                raise ValueError(f"Identity '{name}' not found in the database.")

            mask = labels != name
            new_labels = labels[mask]
            new_templates = database["templates"][mask]
            new_quality = database["template_quality_scores"][mask]
            new_sources = database["template_source_images"][mask]
            new_identity_names = database["identity_names"][database["identity_names"] != name]

            meta = self._load_or_create_meta(db_path)
            meta.get("images_per_identity", {}).pop(name, None)
            self.save_database(
                str(db_path),
                new_identity_names,
                new_templates,
                new_labels,
                new_quality,
                new_sources,
                float(database["global_threshold"]),
            )
            database = self.load_database(str(db_path))
            self._write_metadata(db_path, database, meta)

        return {
            "name": name,
            "num_identities": len(new_identity_names),
            "num_templates": len(new_labels),
        }

    @staticmethod
    def compute_identity_scores(
        database: Dict[str, Any],
        feature: np.ndarray,
        alpha: float = DEFAULT_ALPHA,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, object]]:
        top_k = max(1, int(top_k))
        templates = database["templates"]
        labels = database["template_labels"]
        scores = np.dot(templates, feature)
        candidates: List[Dict[str, object]] = []

        for label in database["identity_names"]:
            person_scores = scores[labels == label]
            if len(person_scores) == 0:
                continue
            top_scores = sorted((float(score) for score in person_scores), reverse=True)[:top_k]
            mean_top_k_score = float(np.mean(top_scores))
            identity_score = alpha * top_scores[0] + (1.0 - alpha) * mean_top_k_score
            candidates.append(
                {
                    "identity": str(label),
                    "score": float(identity_score),
                    "top1_score": float(top_scores[0]),
                    "mean_top_k_score": mean_top_k_score,
                    "top_template_scores": top_scores,
                    "num_templates": int(len(person_scores)),
                }
            )

        if not candidates:
            raise RuntimeError("Face database is empty.")
        return sorted(candidates, key=lambda item: float(item["score"]), reverse=True)

    @staticmethod
    def score_templates(
        labels: np.ndarray, embeddings: np.ndarray, feature: np.ndarray,
    ) -> Dict[str, object]:
        database = {
            "identity_names": np.unique(labels),
            "templates": embeddings,
            "template_labels": labels,
        }
        best = FaceRecognitionSystem.compute_identity_scores(database, feature)[0]
        return {
            "best_match": best["identity"],
            "similarity": float(best["score"]),
            "best_template_similarity": float(best["top1_score"]),
            "top_template_scores": best["top_template_scores"],
        }

    def recognize(
        self,
        image_path: str,
        database_path: str,
        threshold: Optional[float] = None,
        alpha: float = DEFAULT_ALPHA,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, object]:
        database = self.load_database(database_path)
        feature, box, landmarks = self.extract_feature(image_path)
        if feature is None:
            return {
                "image_path": image_path,
                "success": False,
                "message": "No valid face detected",
            }

        global_threshold = float(database["global_threshold"] if threshold is None else threshold)
        candidates = self.compute_identity_scores(database, feature, alpha=alpha, top_k=top_k)
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        best_score = float(best["score"])
        second_score = float(second["score"]) if second is not None else -float("inf")

        # Decision logic: reject weak matches; otherwise accept the highest-scoring identity.
        predicted_name = "stranger" if best_score < global_threshold else str(best["identity"])

        return {
            "image_path": image_path,
            "success": True,
            "predicted_identity": predicted_name,
            "best_match": best["identity"],
            "best_score": round(best_score, 4),
            "similarity": round(best_score, 4),
            "second_best_match": second["identity"] if second is not None else None,
            "second_best_score": round(second_score, 4) if second is not None else None,
            "global_threshold": global_threshold,
            "top_candidates": [
                {
                    **candidate,
                    "score": round(float(candidate["score"]), 4),
                    "top1_score": round(float(candidate["top1_score"]), 4),
                    "mean_top_k_score": round(float(candidate["mean_top_k_score"]), 4),
                    "top_template_scores": [
                        round(float(score), 4) for score in candidate["top_template_scores"]
                    ],
                }
                for candidate in candidates[:5]
            ],
            "database_version": database["database_version"],
            "threshold": global_threshold,
            "is_known": predicted_name != "stranger",
            "face_box": box.tolist() if box is not None else None,
            "landmarks": landmarks.tolist() if landmarks is not None else None,
        }

    def visualize_result(
        self, image_path: str, result: Dict[str, object], save_path: Optional[str] = None
    ) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        if result.get("success") and result.get("face_box") is not None:
            x1, y1, x2, y2 = [int(v) for v in result["face_box"]]
            color = (0, 255, 0) if result.get("is_known") else (255, 0, 0)
            label = f"{result['predicted_identity']} : {result['similarity']:.4f}"

            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            text_y = max(0, y1 - 18)
            draw.text((x1, text_y), label, fill=color)

            if result.get("landmarks") is not None:
                for x, y in result["landmarks"]:
                    draw.ellipse((int(x) - 2, int(y) - 2, int(x) + 2, int(y) + 2), fill=(255, 255, 0))

        if save_path:
            image.save(save_path)

        return image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MTCNN + VGGFace2 face recognition system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser("build", help="Build face template database")
    build_cmd.add_argument("--dataset_dir", required=True, help="Root directory of the face dataset")
    build_cmd.add_argument("--output", default="face_database.npz", help="Output database file path")
    build_cmd.add_argument("--device", default=None, help="mps, cuda:0, or cpu")
    build_cmd.add_argument("--global_threshold", type=float, default=DEFAULT_GLOBAL_THRESHOLD)

    infer_cmd = subparsers.add_parser("recognize", help="Recognize a face from an input image")
    infer_cmd.add_argument("--image", required=True, help="Input face image")
    infer_cmd.add_argument("--database", default="face_database.npz", help="Face template database file")
    infer_cmd.add_argument("--threshold", type=float, default=None, help="Override database global threshold")
    infer_cmd.add_argument("--global_threshold", type=float, default=None, help="Alias for --threshold")
    infer_cmd.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Top-1 score weight")
    infer_cmd.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of top template scores to average")
    infer_cmd.add_argument("--save_vis", default=None, help="Optional output path for visualization image")
    infer_cmd.add_argument("--device", default=None, help="mps, cuda:0, or cpu")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    system = FaceRecognitionSystem(device=args.device)

    if args.command == "build":
        templates = system.build_database(
            args.dataset_dir,
            args.output,
            global_threshold=args.global_threshold,
        )
        num_templates = sum(len(person_templates) for person_templates in templates.values())
        print(f"Database built successfully: {len(templates)} identities and {num_templates} templates generated.")
        print(f"Template database file: {Path(args.output).resolve()}")
        print(f"Metadata file: {Path(args.output).with_suffix('.json').resolve()}")
    elif args.command == "recognize":
        threshold = args.threshold if args.threshold is not None else args.global_threshold
        result = system.recognize(
            args.image,
            args.database,
            threshold=threshold,
            alpha=args.alpha,
            top_k=args.top_k,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.save_vis:
            system.visualize_result(args.image, result, args.save_vis)
            print(f"Visualization result saved to: {Path(args.save_vis).resolve()}")


if __name__ == "__main__":
    main()
