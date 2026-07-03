import argparse
import json
from pathlib import Path

from face_recognition_system import FaceRecognitionSystem


def evaluate(test_dir: str, database_path: str, threshold: float, device: str | None = None) -> dict:
    system = FaceRecognitionSystem(device=device)
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    total = 0
    correct = 0
    error_cases = []

    for person_dir in sorted(test_path.iterdir()):
        if not person_dir.is_dir():
            continue

        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            total += 1
            result = system.recognize(str(image_path), database_path, threshold=threshold)

            if not result.get("success"):
                error_cases.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "true_identity": person_dir.name,
                        "predicted_identity": "no_face_detected",
                    }
                )
                continue

            if result.get("predicted_identity") == person_dir.name:
                correct += 1
            else:
                error_cases.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "true_identity": person_dir.name,
                        "predicted_identity": result.get("predicted_identity"),
                    }
                )

    accuracy = round(correct / total, 4) if total else 0.0
    return {
        "accuracy": accuracy,
        "error_cases": error_cases,
    }


def write_error_report(error_cases: list[dict], output_path: str) -> None:
    lines = ["Face Recognition Error Cases", ""]
    if not error_cases:
        lines.append("No error cases found.")
    else:
        for index, case in enumerate(error_cases, start=1):
            lines.append(
                f"{index}. image={case['image_path']}, true_identity={case['true_identity']}, "
                f"predicted_identity={case['predicted_identity']}"
            )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate face recognition system on test split")
    parser.add_argument("--test_dir", default="dataset/test", help="Test split directory")
    parser.add_argument("--database", default="face_database.npz", help="Template database path")
    parser.add_argument("--threshold", type=float, default=0.75, help="Recognition threshold")
    parser.add_argument("--device", default=None, help="cpu or cuda:0")
    parser.add_argument("--output", default=None, help="Optional JSON file to save evaluation results")
    parser.add_argument(
        "--error_output",
        default="evaluate_result.txt",
        help="Text file used to save error recognition cases",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args.test_dir, args.database, args.threshold, device=args.device)
    write_error_report(result["error_cases"], args.error_output)
    print(f"Overall accuracy: {result['accuracy']:.4f}")
    print(f"Error cases saved to: {Path(args.error_output).resolve()}")

    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Evaluation summary saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
