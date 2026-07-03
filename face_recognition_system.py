import argparse
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, ImageDraw


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


class FaceRecognitionSystem:
    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # MTCNN completes face detection, landmark localization, alignment, and normalization.
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=True,
            device=self.device,
        )

        # CNN embedding network pretrained on VGGFace2.
        self.feature_extractor = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # Serialise database write operations to prevent concurrent corruption.
        self._db_lock = threading.Lock()

    def detect_align_face(
        self, image_path: str
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[np.ndarray]]:
        image = Image.open(image_path).convert("RGB")
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        if boxes is None or probs is None or len(boxes) == 0:
            return None, None, None

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
            return None, None, None

        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)

        return face_tensor.to(self.device), boxes[best_index], landmarks[best_index]

    @torch.no_grad()
    def extract_feature(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        aligned_face, box, landmarks = self.detect_align_face(image_path)
        if aligned_face is None:
            return None, None, None

        embedding = self.feature_extractor(aligned_face)
        feature = embedding.squeeze(0).cpu().numpy().astype(np.float32)
        feature = l2_normalize(feature)
        return feature, box, landmarks

    def build_database(self, dataset_dir: str, output_path: str) -> Dict[str, np.ndarray]:
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_dir}")

        templates: Dict[str, np.ndarray] = {}
        stats: Dict[str, int] = {}

        for person_dir in sorted(dataset_path.iterdir()):
            if not person_dir.is_dir():
                continue

            person_features: List[np.ndarray] = []
            for image_path in sorted(person_dir.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                feature, _, _ = self.extract_feature(str(image_path))
                if feature is not None:
                    person_features.append(feature)

            if not person_features:
                continue

            mean_feature = np.mean(np.vstack(person_features), axis=0)
            templates[person_dir.name] = l2_normalize(mean_feature).astype(np.float32)
            stats[person_dir.name] = len(person_features)

        if not templates:
            raise RuntimeError("No valid faces were extracted from the dataset.")

        embeddings = np.vstack(list(templates.values())).astype(np.float32)
        labels = np.array(list(templates.keys()))
        np.savez(output_path, labels=labels, embeddings=embeddings)

        meta_path = Path(output_path).with_suffix(".json")
        meta = {
            "dataset_dir": str(dataset_path.resolve()),
            "num_identities": len(labels),
            "images_per_identity": stats,
            "feature_dim": int(embeddings.shape[1]),
            "model": "InceptionResnetV1(pretrained=vggface2)",
            "detector": "MTCNN",
            "similarity": "cosine",
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return templates

    # ------------------------------------------------------------------
    # Incremental database operations
    # ------------------------------------------------------------------

    def _extract_features_from_dir(self, person_dir: Path) -> List[np.ndarray]:
        """Extract face features from all valid images in a directory."""
        person_features: List[np.ndarray] = []
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            feature, _, _ = self.extract_feature(str(image_path))
            if feature is not None:
                person_features.append(feature)
        return person_features

    def _write_database(
        self, db_path: Path, labels: np.ndarray, embeddings: np.ndarray,
        meta: dict, meta_path: Path,
    ) -> None:
        """Persist labels/embeddings to .npz and metadata to .json."""
        np.savez(str(db_path), labels=labels, embeddings=embeddings.astype(np.float32))
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def _load_or_create_meta(self, db_path: Path) -> dict:
        """Return existing metadata dict, or a sensible default for a new database."""
        meta_path = db_path.with_suffix(".json")
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "dataset_dir": "",
            "num_identities": 0,
            "images_per_identity": {},
            "feature_dim": 512,
            "model": "InceptionResnetV1(pretrained=vggface2)",
            "detector": "MTCNN",
            "similarity": "cosine",
        }

    def add_identity(
        self, name: str, person_dir: str, database_path: str,
    ) -> dict:
        """Extract features for a **new** identity and append its template to the
        database.

        ``person_dir`` must be a directory containing one or more face images
        (``.jpg`` / ``.png`` / ``.bmp``).  If the database file does not exist
        yet it is created from scratch.

        Returns a summary dict with keys ``name``, ``num_images``,
        ``num_identities``.

        Raises ``ValueError`` when *name* already exists in the database.
        """
        person_path = Path(person_dir)
        if not person_path.is_dir():
            raise FileNotFoundError(f"Person directory not found: {person_dir}")

        db_path = Path(database_path)
        meta_path = db_path.with_suffix(".json")

        with self._db_lock:
            person_features = self._extract_features_from_dir(person_path)
            if not person_features:
                raise RuntimeError(f"No valid faces found in {person_dir}")

            mean_feature = np.mean(np.vstack(person_features), axis=0)
            normalized = l2_normalize(mean_feature).astype(np.float32)

            meta = self._load_or_create_meta(db_path)

            if db_path.exists():
                labels, embeddings = self.load_database(str(db_path))
                if name in labels:
                    raise ValueError(
                        f"Identity '{name}' already exists in the database. "
                        f"Use update_identity() to add more images."
                    )
                new_labels = np.append(labels, name)
                new_embeddings = np.vstack([embeddings, normalized.reshape(1, -1)])
                meta["dataset_dir"] = meta.get("dataset_dir") or str(person_path.parent.resolve())
            else:
                new_labels = np.array([name])
                new_embeddings = normalized.reshape(1, -1)
                meta["dataset_dir"] = str(person_path.parent.resolve())
                meta["feature_dim"] = int(new_embeddings.shape[1])

            meta["num_identities"] = len(new_labels)
            meta.setdefault("images_per_identity", {})[name] = len(person_features)

            self._write_database(db_path, new_labels, new_embeddings, meta, meta_path)

        return {
            "name": name,
            "num_images": len(person_features),
            "num_identities": len(new_labels),
        }

    def update_identity(
        self, name: str, person_dir: str, database_path: str,
    ) -> dict:
        """Re-compute the template for an **existing** identity using all images
        currently present in ``person_dir`` (including any newly added ones).

        The database must already exist and contain *name*.

        Returns a summary dict with keys ``name``, ``num_images``,
        ``num_identities``.

        Raises ``ValueError`` when *name* is not found in the database.
        """
        person_path = Path(person_dir)
        if not person_path.is_dir():
            raise FileNotFoundError(f"Person directory not found: {person_dir}")

        db_path = Path(database_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {database_path}")

        meta_path = db_path.with_suffix(".json")

        with self._db_lock:
            labels, embeddings = self.load_database(str(db_path))
            indices = np.where(labels == name)[0]
            if len(indices) == 0:
                raise ValueError(
                    f"Identity '{name}' not found in the database. "
                    f"Use add_identity() to register a new identity."
                )
            idx = indices[0]

            person_features = self._extract_features_from_dir(person_path)
            if not person_features:
                raise RuntimeError(f"No valid faces found in {person_dir}")

            mean_feature = np.mean(np.vstack(person_features), axis=0)
            normalized = l2_normalize(mean_feature).astype(np.float32)
            embeddings[idx] = normalized

            meta = self._load_or_create_meta(db_path)
            meta.setdefault("images_per_identity", {})[name] = len(person_features)

            self._write_database(db_path, labels, embeddings, meta, meta_path)

        return {
            "name": name,
            "num_images": len(person_features),
            "num_identities": len(labels),
        }

    def remove_identity(self, name: str, database_path: str) -> dict:
        """Remove an identity and its template from the database.

        Returns a summary dict with keys ``name``, ``num_identities``.

        Raises ``ValueError`` when *name* is not found in the database.
        """
        db_path = Path(database_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {database_path}")

        meta_path = db_path.with_suffix(".json")

        with self._db_lock:
            labels, embeddings = self.load_database(str(db_path))
            indices = np.where(labels == name)[0]
            if len(indices) == 0:
                raise ValueError(f"Identity '{name}' not found in the database.")

            mask = labels != name
            new_labels = labels[mask]
            new_embeddings = embeddings[mask]

            meta = self._load_or_create_meta(db_path)
            meta["num_identities"] = len(new_labels)
            meta.get("images_per_identity", {}).pop(name, None)

            self._write_database(db_path, new_labels, new_embeddings, meta, meta_path)

        return {"name": name, "num_identities": len(new_labels)}

    @staticmethod
    def load_database(database_path: str) -> Tuple[np.ndarray, np.ndarray]:
        if not Path(database_path).exists():
            raise FileNotFoundError(f"Database file does not exist: {database_path}")
        data = np.load(database_path, allow_pickle=True)
        labels = data["labels"]
        embeddings = data["embeddings"].astype(np.float32)
        return labels, embeddings

    def score_image(
        self, image_path: str, database_path: str
    ) -> Dict[str, object]:
        """Extract the face feature for ``image_path`` and score it against every
        template in the database, returning the best match and its similarity
        WITHOUT applying a decision threshold.

        This is the threshold-free core of :meth:`recognize`. Threshold-tuning
        code uses it so that feature extraction runs once per image while many
        decision thresholds can be evaluated cheaply afterwards.
        """
        labels, embeddings = self.load_database(database_path)
        feature, box, landmarks = self.extract_feature(image_path)
        if feature is None:
            return {
                "image_path": image_path,
                "success": False,
                "message": "No valid face detected",
            }

        scores = np.dot(embeddings, feature)
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        return {
            "image_path": image_path,
            "success": True,
            "best_match": str(labels[best_index]),
            "similarity": best_score,
            "face_box": box.tolist() if box is not None else None,
            "landmarks": landmarks.tolist() if landmarks is not None else None,
        }

    def recognize(self, image_path: str, database_path: str, threshold: float = 0.48) -> Dict[str, object]:
        labels, embeddings = self.load_database(database_path)
        feature, box, landmarks = self.extract_feature(image_path)
        if feature is None:
            return {
                "image_path": image_path,
                "success": False,
                "message": "No valid face detected",
            }

        scores = np.dot(embeddings, feature)
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        predicted_name = str(labels[best_index]) if best_score >= threshold else "stranger"

        return {
            "image_path": image_path,
            "success": True,
            "predicted_identity": predicted_name,
            "best_match": str(labels[best_index]),
            "similarity": round(best_score, 4),
            "threshold": threshold,
            "is_known": bool(best_score >= threshold),
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
    build_cmd.add_argument("--device", default=None, help="cpu or cuda:0")

    infer_cmd = subparsers.add_parser("recognize", help="Recognize a face from an input image")
    infer_cmd.add_argument("--image", required=True, help="Input face image")
    infer_cmd.add_argument("--database", default="face_database.npz", help="Face template database file")
    infer_cmd.add_argument("--threshold", type=float, default=0.75, help="Unknown-person decision threshold")
    infer_cmd.add_argument("--save_vis", default=None, help="Optional output path for visualization image")
    infer_cmd.add_argument("--device", default=None, help="cpu or cuda:0")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    system = FaceRecognitionSystem(device=args.device)

    if args.command == "build":
        templates = system.build_database(args.dataset_dir, args.output)
        print(f"Database built successfully: {len(templates)} identities and {num_templates} templates generated.")
        print(f"Template database file: {Path(args.output).resolve()}")
        print(f"Metadata file: {Path(args.output).with_suffix('.json').resolve()}")
    elif args.command == "recognize":
        result = system.recognize(args.image, args.database, threshold=args.threshold)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.save_vis:
            system.visualize_result(args.image, result, args.save_vis)
            print(f"Visualization result saved to: {Path(args.save_vis).resolve()}")

if __name__ == "__main__":
    main()
