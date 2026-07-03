import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
STRANGER_LABEL = "stranger"
NO_FACE_LABEL = "no_face_detected"
FAR_CONSTRAINTS = (0.01, 0.03, 0.05, 0.1)
PRIMARY_FAR_CONSTRAINT = FAR_CONSTRAINTS[0]
ROC_X_MIN = 0.01
ROC_X_MAX = 0.5
ROC_Y_MIN = 0.88
ROC_Y_MAX = 0.95
ROC_EPSILON = 1e-5
THRESHOLD_START = 0.3
THRESHOLD_END = 1.0
THRESHOLD_STEP = 0.001
CONSTRAINT_COLORS = ("red", "purple", "darkgreen", "orange")
RESULTS_DIR = Path("results")


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def round_metric(value: float) -> float:
    return round(float(value), 6)


def ensure_parent_dir(output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def resolve_result_output_path(output_path: Optional[str], default_name: str) -> str:
    path = Path(output_path) if output_path else Path(default_name)
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == RESULTS_DIR.name:
        return str(path)
    return str(RESULTS_DIR / path)


def load_database_metadata(metadata_path: Optional[str]) -> Dict[str, Any]:
    if not metadata_path:
        return {}

    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Database metadata file does not exist: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def iter_test_images(test_dir: str) -> Iterable[Tuple[str, Path]]:
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    for person_dir in sorted(test_path.iterdir()):
        if not person_dir.is_dir():
            continue
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                yield person_dir.name, image_path


def build_thresholds(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("threshold step must be positive")
    if end < start:
        raise ValueError("threshold end must be greater than or equal to threshold start")

    thresholds: List[float] = []
    current = start
    epsilon = step / 10.0
    while current <= end + epsilon:
        thresholds.append(round(float(current), 6))
        current += step
    return thresholds


def score_image(
    system: Any,
    image_path: Path,
    database: Dict[str, Any],
) -> Dict[str, Any]:
    if hasattr(system, "extract_template"):
        feature, box, landmarks, confidence = system.extract_template(str(image_path))
    else:
        feature, box, landmarks = system.extract_feature(str(image_path))
        confidence = None

    if feature is None:
        return {
            "success": False,
            "best_match": None,
            "similarity": None,
            "message": "No valid face detected",
            "face_box": None,
            "landmarks": None,
            "detection_confidence": None,
        }

    candidates = system.compute_identity_scores(database, feature)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    return {
        "success": True,
        "best_match": best["identity"],
        "similarity": float(best["score"]),
        "best_template_similarity": float(best["top1_score"]),
        "top_template_scores": best["top_template_scores"],
        "second_best_match": second["identity"] if second is not None else None,
        "second_best_score": float(second["score"]) if second is not None else None,
        "message": None,
        "face_box": box.tolist() if box is not None else None,
        "landmarks": landmarks.tolist() if landmarks is not None else None,
        "detection_confidence": float(confidence) if confidence is not None else None,
    }


def collect_image_scores(
    system: Any,
    test_dir: str,
    database: Dict[str, Any],
    known_identities: set[str],
) -> List[Dict[str, Any]]:
    samples = list(iter_test_images(test_dir))
    if not samples:
        raise RuntimeError(f"No test images found in: {test_dir}")

    scored_samples: List[Dict[str, Any]] = []
    for index, (true_identity, image_path) in enumerate(samples, start=1):
        scored = score_image(system, image_path, database)
        scored_samples.append(
            {
                "index": index,
                "image_path": str(image_path.resolve()),
                "true_identity": true_identity,
                "is_database_identity": true_identity in known_identities,
                "success": scored["success"],
                "best_match": scored["best_match"],
                "similarity": scored["similarity"],
                "best_template_similarity": scored.get("best_template_similarity"),
                "top_template_scores": scored.get("top_template_scores"),
                "second_best_match": scored.get("second_best_match"),
                "second_best_score": scored.get("second_best_score"),
                "message": scored["message"],
                "face_box": scored["face_box"],
                "landmarks": scored["landmarks"],
                "detection_confidence": scored["detection_confidence"],
            }
        )
    return scored_samples


def predicted_identity_for_threshold(sample: Dict[str, Any], threshold: float) -> str:
    if not sample["success"] or sample["similarity"] is None:
        return NO_FACE_LABEL
    return str(sample["best_match"]) if float(sample["similarity"]) >= threshold else STRANGER_LABEL


def calculate_metrics(
    scored_samples: Sequence[Dict[str, Any]],
    known_identities: set[str],
    threshold: float,
) -> Dict[str, Any]:
    known_total = 0
    known_correct = 0
    known_rejected = 0
    known_misidentified = 0
    stranger_total = 0
    stranger_correct_reject = 0
    stranger_false_accept = 0

    for sample in scored_samples:
        true_label = str(sample["true_identity"])
        pred_label = predicted_identity_for_threshold(sample, threshold)

        if true_label in known_identities:
            known_total += 1
            if pred_label == true_label:
                known_correct += 1
            elif pred_label in {STRANGER_LABEL, NO_FACE_LABEL}:
                known_rejected += 1
            else:
                known_misidentified += 1
        else:
            stranger_total += 1
            if pred_label in {STRANGER_LABEL, NO_FACE_LABEL}:
                stranger_correct_reject += 1
            else:
                stranger_false_accept += 1

    total = known_total + stranger_total
    far = safe_divide(stranger_false_accept, stranger_total)
    recall = safe_divide(known_correct, known_total)
    frr = safe_divide(known_rejected, known_total)
    open_set_accuracy = safe_divide(known_correct + stranger_correct_reject, total)
    precision = safe_divide(
        known_correct,
        known_correct + known_misidentified + stranger_false_accept,
    )
    f1_score = safe_divide(2 * precision * recall, precision + recall)

    return {
        "threshold": round_metric(threshold),
        "total_samples": total,
        "known_total": known_total,
        "stranger_total": stranger_total,
        "known_correct": known_correct,
        "known_rejected": known_rejected,
        "known_misidentified": known_misidentified,
        "stranger_correct_reject": stranger_correct_reject,
        "stranger_false_accept": stranger_false_accept,
        "far": round_metric(far),
        "recall_tar": round_metric(recall),
        "frr": round_metric(frr),
        "open_set_accuracy": round_metric(open_set_accuracy),
        "precision": round_metric(precision),
        "f1_score": round_metric(f1_score),
    }


def scan_thresholds(
    scored_samples: Sequence[Dict[str, Any]],
    known_identities: set[str],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    return [calculate_metrics(scored_samples, known_identities, threshold) for threshold in thresholds]


def exact_far(metrics: Dict[str, Any]) -> float:
    return safe_divide(metrics["stranger_false_accept"], metrics["stranger_total"])


def exact_recall_tar(metrics: Dict[str, Any]) -> float:
    return safe_divide(metrics["known_correct"], metrics["known_total"])


def select_threshold(threshold_metrics: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[str]]:
    eligible = [metrics for metrics in threshold_metrics if exact_far(metrics) <= 0.001]
    if eligible:
        return max(
            eligible,
            key=lambda metrics: (
                exact_recall_tar(metrics),
                -exact_far(metrics),
                metrics["threshold"],
            ),
        ), None

    selected = min(
        threshold_metrics,
        key=lambda metrics: (
            exact_far(metrics),
            -exact_recall_tar(metrics),
            -metrics["threshold"],
        ),
    )
    warning = (
        "WARNING: No scanned threshold satisfied FAR <= 0.001. "
        "Selected the threshold with the lowest FAR instead."
    )
    return selected, warning


def select_metrics_for_far_constraint(
    threshold_metrics: Sequence[Dict[str, Any]],
    far_constraint: float,
) -> Dict[str, Any]:
    eligible = [metrics for metrics in threshold_metrics if exact_far(metrics) <= far_constraint]
    if eligible:
        selected = max(
            eligible,
            key=lambda metrics: (
                exact_recall_tar(metrics),
                -exact_far(metrics),
                metrics["threshold"],
            ),
        )
        return {
            **selected,
            "far_constraint": far_constraint,
            "constraint_satisfied": True,
            "selection_warning": None,
        }

    selected = min(
        threshold_metrics,
        key=lambda metrics: (
            exact_far(metrics),
            -exact_recall_tar(metrics),
            -metrics["threshold"],
        ),
    )
    return {
        **selected,
        "far_constraint": far_constraint,
        "constraint_satisfied": False,
        "selection_warning": (
            f"No scanned threshold satisfied FAR <= {far_constraint:g}; "
            "selected the threshold with the lowest FAR instead."
        ),
    }


def select_metrics_for_far_constraints(
    threshold_metrics: Sequence[Dict[str, Any]],
    far_constraints: Tuple[float, ...] = FAR_CONSTRAINTS,
) -> List[Dict[str, Any]]:
    return [
        select_metrics_for_far_constraint(threshold_metrics, far_constraint)
        for far_constraint in far_constraints
    ]


def constrained_objective_score(metrics: Dict[str, Any]) -> float:
    return round_metric(exact_recall_tar(metrics) if exact_far(metrics) <= 0.001 else 0.0)


def attach_predictions(
    scored_samples: Sequence[Dict[str, Any]],
    known_identities: set[str],
    threshold: float,
) -> List[Dict[str, Any]]:
    samples_with_predictions: List[Dict[str, Any]] = []
    for sample in scored_samples:
        predicted_identity = predicted_identity_for_threshold(sample, threshold)
        true_identity = str(sample["true_identity"])
        is_database_identity = true_identity in known_identities
        is_exact_match = predicted_identity == true_identity
        is_open_set_correct = (
            is_exact_match
            if is_database_identity
            else predicted_identity in {STRANGER_LABEL, NO_FACE_LABEL}
        )
        samples_with_predictions.append(
            {
                **sample,
                "predicted_identity": predicted_identity,
                "is_known_prediction": predicted_identity not in {STRANGER_LABEL, NO_FACE_LABEL},
                "is_exact_match": is_exact_match,
                "is_open_set_correct": is_open_set_correct,
                "similarity": round_metric(sample["similarity"]) if sample["similarity"] is not None else None,
            }
        )
    return samples_with_predictions


def write_json_report(result: Dict[str, Any], output_path: str) -> None:
    ensure_parent_dir(output_path)
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def write_samples_csv(samples: Sequence[Dict[str, Any]], output_path: str) -> None:
    fieldnames = [
        "index",
        "image_path",
        "true_identity",
        "predicted_identity",
        "best_match",
        "similarity",
        "second_best_match",
        "second_best_score",
        "best_template_similarity",
        "top_template_scores",
        "success",
        "detection_confidence",
        "is_known_prediction",
        "is_database_identity",
        "is_exact_match",
        "is_open_set_correct",
        "message",
    ]
    ensure_parent_dir(output_path)
    with Path(output_path).open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({field: sample.get(field) for field in fieldnames})


def write_metrics_summary_csv(metrics: Dict[str, Any], output_path: str) -> None:
    ensure_parent_dir(output_path)
    with Path(output_path).open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        for key in [
            "threshold",
            "far",
            "recall_tar",
            "frr",
            "open_set_accuracy",
            "precision",
            "f1_score",
            "total_samples",
            "known_total",
            "known_correct",
            "known_rejected",
            "known_misidentified",
            "stranger_total",
            "stranger_correct_reject",
            "stranger_false_accept",
        ]:
            writer.writerow([key, metrics[key]])


def write_threshold_scan_csv(threshold_metrics: Sequence[Dict[str, Any]], output_path: str) -> None:
    fieldnames = [
        "threshold",
        "known_correct",
        "known_rejected",
        "known_misidentified",
        "stranger_correct_reject",
        "stranger_false_accept",
        "far",
        "recall_tar",
        "frr",
        "open_set_accuracy",
        "precision",
        "f1_score",
    ]
    ensure_parent_dir(output_path)
    with Path(output_path).open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: metrics[field] for field in fieldnames} for metrics in threshold_metrics)


def write_far_constraint_metrics_csv(metrics_rows: Sequence[Dict[str, Any]], output_path: str) -> None:
    fieldnames = [
        "far_constraint",
        "constraint_satisfied",
        "threshold",
        "far",
        "recall_tar",
        "frr",
        "open_set_accuracy",
        "precision",
        "f1_score",
        "known_total",
        "known_correct",
        "known_rejected",
        "known_misidentified",
        "stranger_total",
        "stranger_correct_reject",
        "stranger_false_accept",
        "selection_warning",
    ]
    ensure_parent_dir(output_path)
    with Path(output_path).open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_error_report(error_cases: Sequence[Dict[str, Any]], output_path: str) -> None:
    lines = ["Face Recognition Error Cases", ""]
    if not error_cases:
        lines.append("No error cases found.")
    else:
        for case in error_cases:
            lines.append(
                f"{case['index']}. image={case['image_path']}, true_identity={case['true_identity']}, "
                f"predicted_identity={case['predicted_identity']}, best_match={case['best_match']}, "
                f"similarity={case['similarity']}, message={case['message']}"
            )
    ensure_parent_dir(output_path)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def plot_roc_curve(
    threshold_metrics: Sequence[Dict[str, Any]],
    output_path: str,
    far_constraint_metrics: Optional[Sequence[Dict[str, Any]]] = None,
    log_scale: bool = False,
) -> None:
    far_values = [max(float(metrics["far"]), ROC_EPSILON) for metrics in threshold_metrics]
    recall_values = [metrics["recall_tar"] for metrics in threshold_metrics]
    if far_constraint_metrics is None:
        far_constraint_metrics = select_metrics_for_far_constraints(threshold_metrics)

    plt.figure(figsize=(8, 6))
    plt.plot(
        far_values,
        recall_values,
        linewidth=1.8,
        color="#1f77b4",
        label="ROC",
    )
    for index, far_constraint in enumerate(FAR_CONSTRAINTS):
        plt.axvline(
            far_constraint,
            color=CONSTRAINT_COLORS[index % len(CONSTRAINT_COLORS)],
            linestyle="--",
            linewidth=1.3,
            label=f"FAR = {far_constraint:g}",
        )

    for index, metrics in enumerate(far_constraint_metrics):
        display_far = max(float(metrics["far"]), ROC_EPSILON)
        plt.scatter(
            [display_far],
            [metrics["recall_tar"]],
            marker="*",
            s=150,
            color=CONSTRAINT_COLORS[index % len(CONSTRAINT_COLORS)],
            edgecolors="black",
            linewidths=0.6,
            label=f"Best FAR <= {metrics['far_constraint']:g}: t={metrics['threshold']:.2f}",
            zorder=4,
        )
        plt.annotate(
            f"t={metrics['threshold']:.2f}",
            xy=(display_far, metrics["recall_tar"]),
            xytext=(8, 8 + index * 8),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )

    plt.xlabel("FAR")
    plt.ylabel("Recall / TAR")
    scale_label = "Log-scale" if log_scale else "Linear-scale"
    plt.title(f"Open-set Face Recognition ROC Curve ({scale_label})")
    if log_scale:
        plt.xscale("log")
    plt.xlim(ROC_X_MIN, ROC_X_MAX)
    plt.ylim(ROC_Y_MIN, ROC_Y_MAX)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curves(
    threshold_metrics: Sequence[Dict[str, Any]],
    linear_output_path: str,
    log_output_path: str,
    far_constraint_metrics: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    plot_roc_curve(
        threshold_metrics,
        linear_output_path,
        far_constraint_metrics=far_constraint_metrics,
        log_scale=False,
    )
    plot_roc_curve(
        threshold_metrics,
        log_output_path,
        far_constraint_metrics=far_constraint_metrics,
        log_scale=True,
    )


def evaluate(
    test_dir: str,
    database_path: str,
    metadata_path: Optional[str] = "face_database.json",
    device: Optional[str] = None,
    threshold_start: float = THRESHOLD_START,
    threshold_end: float = THRESHOLD_END,
    threshold_step: float = THRESHOLD_STEP,
    roc_linear_output: str = "results/ROC_linear.png",
    roc_log_output: str = "results/ROC_log.png",
    include_details: bool = False,
) -> Dict[str, Any]:
    from face_recognition_system import FaceRecognitionSystem

    system = FaceRecognitionSystem(device=device)
    database = system.load_database(database_path)
    metadata = load_database_metadata(metadata_path)
    known_identities = {str(label) for label in database["identity_names"].tolist()}
    thresholds = build_thresholds(threshold_start, threshold_end, threshold_step)
    scored_samples = collect_image_scores(system, test_dir, database, known_identities)
    threshold_metrics = scan_thresholds(scored_samples, known_identities, thresholds)
    far_constraint_metrics = select_metrics_for_far_constraints(threshold_metrics)
    selected_metrics = far_constraint_metrics[0]
    selection_warning = selected_metrics["selection_warning"]
    final_threshold = float(selected_metrics["threshold"])
    samples_with_predictions = attach_predictions(scored_samples, known_identities, final_threshold)
    final_metrics = calculate_metrics(scored_samples, known_identities, final_threshold)
    plot_roc_curves(threshold_metrics, roc_linear_output, roc_log_output, far_constraint_metrics)

    test_identities = sorted({str(sample["true_identity"]) for sample in scored_samples})
    missing_from_database = [identity for identity in test_identities if identity not in known_identities]
    error_cases = [sample for sample in samples_with_predictions if not sample["is_open_set_correct"]]
    valid_embeddings = sum(1 for sample in scored_samples if sample["success"])
    failed_detections = len(scored_samples) - valid_embeddings

    report: Dict[str, Any] = {
        "configuration": {
            "test_dir": str(Path(test_dir).resolve()),
            "database_path": str(Path(database_path).resolve()),
            "metadata_path": str(Path(metadata_path).resolve()) if metadata_path else None,
            "device": str(system.device),
            "threshold_start": threshold_start,
            "threshold_end": threshold_end,
            "threshold_step": threshold_step,
            "threshold_generation": "inclusive stepped scan from threshold_start to threshold_end",
            "roc_epsilon": ROC_EPSILON,
            "roc_x_min": ROC_X_MIN,
            "roc_x_max": ROC_X_MAX,
            "roc_y_min": ROC_Y_MIN,
            "roc_y_max": ROC_Y_MAX,
            "far_constraints": list(FAR_CONSTRAINTS),
            "roc_linear_output": str(Path(roc_linear_output).resolve()),
            "roc_log_output": str(Path(roc_log_output).resolve()),
            "selection_rule": "for each FAR constraint, choose highest Recall/TAR among thresholds satisfying FAR <= constraint; tie by lower FAR, then higher threshold",
            "selection_warning": selection_warning,
        },
        "database": {
            "num_identities": int(len(known_identities)),
            "num_templates": int(len(database["templates"])),
            "feature_dim": int(database["embedding_dim"]),
            "database_version": database["database_version"],
            "metadata": metadata,
        },
        "test_set": {
            "num_identities": len(test_identities),
            "num_samples": len(scored_samples),
            "valid_embeddings": valid_embeddings,
            "failed_detections": failed_detections,
            "feature_dim": int(database["embedding_dim"]),
            "model_name": database["model_name"],
            "test_cache_version": None,
            "identities_missing_from_database": missing_from_database,
        },
        "selected_threshold": final_threshold,
        "metrics": final_metrics,
        "far_constraint_metrics": far_constraint_metrics,
        "threshold_scan": threshold_metrics,
        "error_count": len(error_cases),
    }
    if include_details:
        report["samples"] = samples_with_predictions
        report["error_cases"] = error_cases

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate open-set face recognition on all test images")
    parser.add_argument("--test_dir", default="dataset/test", help="Test split directory")
    parser.add_argument("--database", default="face_database.npz", help="Face template database path")
    parser.add_argument("--metadata", default="face_database.json", help="Face database metadata JSON path")
    parser.add_argument("--device", default=None, help="cpu, mps, or cuda:0")
    parser.add_argument("--threshold_start", type=float, default=THRESHOLD_START, help="First similarity threshold to scan")
    parser.add_argument("--threshold_end", type=float, default=THRESHOLD_END, help="Last similarity threshold to scan")
    parser.add_argument("--threshold_step", type=float, default=THRESHOLD_STEP, help="Similarity threshold scan step")
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "evaluate_summary.json"),
        help="JSON file for aggregate evaluation results",
    )
    parser.add_argument(
        "--summary_output",
        default=str(RESULTS_DIR / "evaluate_metrics_summary.csv"),
        help="CSV file for selected-threshold metrics",
    )
    parser.add_argument(
        "--far_constraints_output",
        default=str(RESULTS_DIR / "evaluate_far_constraints.csv"),
        help="CSV file for FAR-constrained selected-threshold metrics",
    )
    parser.add_argument("--threshold_scan_output", default=None, help="Optional CSV file for all scanned thresholds")
    parser.add_argument(
        "--roc_linear_output",
        default=None,
        help="Linear-scale ROC curve image output path",
    )
    parser.add_argument(
        "--roc_log_output",
        default=str(RESULTS_DIR / "ROC_log.png"),
        help="Log-scale ROC curve image output path",
    )
    parser.add_argument(
        "--roc_output",
        default=None,
        help="Backward-compatible alias for --roc_linear_output",
    )
    parser.add_argument("--include_samples", action="store_true", help="Include per-image results in JSON and samples CSV")
    parser.add_argument("--samples_output", default=None, help="Optional CSV file for per-image results")
    parser.add_argument(
        "--error_output",
        default=None,
        help="Optional text file used to save failed or misclassified cases",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output = resolve_result_output_path(args.output, "evaluate_summary.json")
    summary_output = resolve_result_output_path(args.summary_output, "evaluate_metrics_summary.csv")
    far_constraints_output = resolve_result_output_path(
        args.far_constraints_output,
        "evaluate_far_constraints.csv",
    )
    threshold_scan_output = (
        resolve_result_output_path(args.threshold_scan_output, "evaluate_threshold_scan.csv")
        if args.threshold_scan_output
        else None
    )
    roc_linear_output = resolve_result_output_path(
        args.roc_linear_output or args.roc_output,
        "ROC_linear.png",
    )
    roc_log_output = resolve_result_output_path(args.roc_log_output, "ROC_log.png")
    samples_output = resolve_result_output_path(
        args.samples_output,
        "evaluate_samples.csv",
    )
    error_output = (
        resolve_result_output_path(args.error_output, "evaluate_errors.txt")
        if args.error_output
        else None
    )

    result = evaluate(
        test_dir=args.test_dir,
        database_path=args.database,
        metadata_path=args.metadata,
        device=args.device,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.threshold_step,
        roc_linear_output=roc_linear_output,
        roc_log_output=roc_log_output,
        include_details=args.include_samples or error_output is not None,
    )

    metrics = result["metrics"]
    write_json_report(result, output)
    write_metrics_summary_csv(metrics, summary_output)
    write_far_constraint_metrics_csv(result["far_constraint_metrics"], far_constraints_output)
    if threshold_scan_output:
        write_threshold_scan_csv(result["threshold_scan"], threshold_scan_output)
    if args.include_samples:
        write_samples_csv(result["samples"], samples_output)
    if error_output:
        write_error_report(result["error_cases"], error_output)

    if result["configuration"]["selection_warning"]:
        print(result["configuration"]["selection_warning"])
    print(f"Samples: {metrics['total_samples']}")
    print(f"Image valid embeddings: {result['test_set']['valid_embeddings']}")
    print(f"Image failed detections: {result['test_set']['failed_detections']}")
    print(f"Selected threshold: {metrics['threshold']:.6f}")
    print(f"FAR: {metrics['far']:.6f}")
    print(f"Recall / TAR: {metrics['recall_tar']:.6f}")
    print(f"FRR: {metrics['frr']:.6f}")
    print(
        "Known summary: "
        f"total={metrics['known_total']}, "
        f"correct={metrics['known_correct']}, "
        f"rejected={metrics['known_rejected']}, "
        f"misidentified={metrics['known_misidentified']}"
    )
    print(
        "Stranger summary: "
        f"total={metrics['stranger_total']}, "
        f"correct_reject={metrics['stranger_correct_reject']}, "
        f"false_accept={metrics['stranger_false_accept']}"
    )
    print(f"Open-set accuracy: {metrics['open_set_accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"F1-score: {metrics['f1_score']:.6f}")
    print("FAR-constrained selected thresholds:")
    for row in result["far_constraint_metrics"]:
        warning_suffix = " (constraint not satisfied)" if not row["constraint_satisfied"] else ""
        print(
            f"  FAR <= {row['far_constraint']:.2f}: "
            f"threshold={row['threshold']:.6f}, "
            f"FAR={row['far']:.6f}, "
            f"Recall/TAR={row['recall_tar']:.6f}, "
            f"FRR={row['frr']:.6f}, "
            f"Open-set accuracy={row['open_set_accuracy']:.6f}, "
            f"Precision={row['precision']:.6f}, "
            f"F1={row['f1_score']:.6f}"
            f"{warning_suffix}"
        )
    print(f"Error cases: {result['error_count']}")
    print(f"JSON summary saved to: {Path(output).resolve()}")
    print(f"Metrics summary saved to: {Path(summary_output).resolve()}")
    print(f"FAR constraint metrics saved to: {Path(far_constraints_output).resolve()}")
    print(f"Linear-scale ROC curve saved to: {Path(roc_linear_output).resolve()}")
    print(f"Log-scale ROC curve saved to: {Path(roc_log_output).resolve()}")
    if threshold_scan_output:
        print(f"Threshold scan saved to: {Path(threshold_scan_output).resolve()}")
    if args.include_samples:
        print(f"Sample results saved to: {Path(samples_output).resolve()}")
    if error_output:
        print(f"Error cases saved to: {Path(error_output).resolve()}")


if __name__ == "__main__":
    main()
