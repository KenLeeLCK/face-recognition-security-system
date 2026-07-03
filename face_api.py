import shutil
import tempfile
import json
import re
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from face_recognition_system import FaceRecognitionSystem


app = FastAPI(title="Face Recognition API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
system = FaceRecognitionSystem()


class BuildDatabaseRequest(BaseModel):
    dataset_dir: str
    output: str = "face_database.npz"
    global_threshold: float = 0.75


class RemoveIdentityRequest(BaseModel):
    name: str
    database: str = "face_database.npz"


def sanitize_identity_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\- ]+", "", name).strip().replace(" ", "_")
    if not cleaned:
        raise ValueError("Person name must contain letters, numbers, spaces, hyphens, or underscores.")
    return cleaned


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "message": "face recognition service is running"}


@app.get("/metrics")
def metrics(database: str = "face_database.npz") -> dict:
    database_path = Path(database)
    meta_path = database_path.with_suffix(".json")

    if not meta_path.exists() and not database_path.exists():
        raise HTTPException(status_code=404, detail=f"Metrics file not found: {meta_path}")

    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            database_data = system.load_database(str(database_path))
            meta = {
                "num_identities": len(database_data["identity_names"]),
                "num_templates": len(database_data["templates"]),
                "feature_dim": database_data["embedding_dim"],
                "model": database_data["model_name"],
                "database_version": database_data["database_version"],
                "global_threshold": database_data["global_threshold"],
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "database_path": str(database_path.resolve()) if database_path.exists() else str(database_path),
        "dataset_dir": meta.get("dataset_dir"),
        "num_identities": meta.get("num_identities"),
        "num_templates": meta.get("num_templates"),
        "feature_dim": meta.get("feature_dim"),
        "model": meta.get("model"),
        "detector": meta.get("detector"),
        "similarity": meta.get("similarity"),
        "database_version": meta.get("database_version"),
        "global_threshold": meta.get("global_threshold"),
        "matching": meta.get("matching"),
    }


@app.get("/identities")
def list_identities(database: str = "face_database.npz", query: str = "") -> dict:
    database_path = Path(database)
    if not database_path.exists():
        raise HTTPException(status_code=404, detail=f"Database file not found: {database_path}")

    try:
        database_data = system.load_database(str(database_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    labels = database_data["template_labels"]
    search = query.strip().lower()
    identities = []

    for identity in database_data["identity_names"]:
        name = str(identity)
        if search and search not in name.lower():
            continue

        identities.append({
            "name": name,
            "num_templates": int((labels == name).sum()),
        })

    identities.sort(key=lambda item: item["name"].lower())

    return {
        "success": True,
        "database_path": str(database_path.resolve()),
        "query": query,
        "num_identities": len(identities),
        "identities": identities,
    }


@app.post("/build_database")
def build_database(request: BuildDatabaseRequest) -> dict:
    dataset_path = Path(request.dataset_dir)
    output_path = Path(request.output)

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset directory not found: {dataset_path}")

    try:
        templates = system.build_database(
            str(dataset_path),
            str(output_path),
            global_threshold=request.global_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "success": True,
        "message": "建库完成",
        "num_identities": len(templates),
        "num_templates": sum(len(person_templates) for person_templates in templates.values()),
        "database_path": str(output_path.resolve()),
        "meta_path": str(output_path.with_suffix(".json").resolve()),
    }


@app.post("/remove_identity")
def remove_identity(request: RemoveIdentityRequest) -> dict:
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Identity name is required.")

    try:
        summary = system.remove_identity(name, request.database)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "success": True,
        "message": "Identity removed successfully",
        "database_path": str(Path(request.database).resolve()),
        **summary,
    }


@app.post("/register")
async def register(
    file: UploadFile = File(...),
    person_name: str = Form(...),
    dataset_dir: str = Form("./dataset/train"),
    output: str = Form("face_database.npz"),
) -> dict:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset directory not found: {dataset_path}")

    try:
        safe_name = sanitize_identity_name(person_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp"}:
        raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, .png, and .bmp files are supported.")

    person_dir = dataset_path / safe_name
    person_dir.mkdir(parents=True, exist_ok=True)

    saved_path = person_dir / f"{safe_name}_{uuid4().hex[:8]}{suffix}"

    try:
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        templates = system.build_database(str(dataset_path), output)
    except Exception as exc:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "success": True,
        "message": "Face registered successfully",
        "person_name": safe_name,
        "saved_image": str(saved_path.resolve()),
        "num_identities": len(templates),
        "num_templates": sum(len(person_templates) for person_templates in templates.values()),
        "database_path": str(Path(output).resolve()),
    }


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    database: str = Form("face_database.npz"),
    threshold: Optional[float] = Form(None),
    global_threshold: Optional[float] = Form(None),
    alpha: float = Form(0.7),
    top_k: int = Form(3),
    save_visualization: bool = Form(False),
    visualization_name: Optional[str] = Form(None),
) -> dict:
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    temp_dir = Path(tempfile.mkdtemp(prefix="face_api_"))
    temp_image = temp_dir / f"input{suffix}"

    try:
        with temp_image.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        threshold_override = threshold if threshold is not None else global_threshold
        result = system.recognize(
            str(temp_image),
            database,
            threshold=threshold_override,
            alpha=alpha,
            top_k=top_k,
        )
        response = dict(result)

        if save_visualization and result.get("success"):
            vis_dir = Path("outputs")
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_name = visualization_name or f"{temp_image.stem}_result.jpg"
            vis_path = vis_dir / vis_name
            system.visualize_result(str(temp_image), result, str(vis_path))
            response["visualization_path"] = str(vis_path.resolve())

        return response
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except OSError:
            pass
