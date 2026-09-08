#!/usr/bin/env python3
"""Read-only validation for AdaptFit documentation and contract fixtures.

The validator intentionally uses only the Python standard library so it can run
before the optional development dependencies are installed. It checks structure
and evidence markers; human review remains required for safety and privacy
wording.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SCHEMAS = DOCS / "schemas"
EXAMPLES = DOCS / "examples" / "contracts"
REQUIRED_METADATA = (
    "Status",
    "Authority",
    "Last verified",
    "Source commit",
    "Owner",
    "Supersedes or supports",
    "Review trigger",
)


class ValidationError(Exception):
    """A fixture or documentation validation failure."""


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_value(value: Any, schema: dict[str, Any], path: str = "$", errors: list[str] | None = None) -> list[str]:
    """Validate the small JSON Schema subset used by checked-in fixtures."""

    local_errors = errors if errors is not None else []
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        type_ok = any(
            (kind == "object" and isinstance(value, dict))
            or (kind == "array" and isinstance(value, list))
            or (kind == "string" and isinstance(value, str))
            or (kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
            or (kind == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (kind == "boolean" and isinstance(value, bool))
            or (kind == "null" and value is None)
            for kind in types
        )
        if not type_ok:
            fail(local_errors, f"{path}: expected {expected}, got {type(value).__name__}")
            return local_errors

    if "const" in schema and value != schema["const"]:
        fail(local_errors, f"{path}: expected constant {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        fail(local_errors, f"{path}: expected one of {schema['enum']!r}, got {value!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            fail(local_errors, f"{path}: below minimum {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            fail(local_errors, f"{path}: not above exclusive minimum {schema['exclusiveMinimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            fail(local_errors, f"{path}: above maximum {schema['maximum']}")
    if isinstance(value, str) and "pattern" in schema and re.search(schema["pattern"], value) is None:
        fail(local_errors, f"{path}: does not match pattern {schema['pattern']!r}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            fail(local_errors, f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            fail(local_errors, f"{path}: more than {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(value):
                validate_value(item, schema["items"], f"{path}[{index}]", local_errors)
    if isinstance(value, dict):
        for required in schema.get("required", []):
            if required not in value:
                fail(local_errors, f"{path}: missing required property {required!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    fail(local_errors, f"{path}: unexpected property {key!r}")
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, dict):
                validate_value(value[key], child_schema, f"{path}.{key}", local_errors)
    return local_errors


def read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(errors, f"{path.relative_to(ROOT)}: invalid JSON: {exc}")
        return None
    if not isinstance(value, dict):
        fail(errors, f"{path.relative_to(ROOT)}: top-level JSON value must be an object")
        return None
    return value


def validate_metadata(errors: list[str]) -> None:
    markdown = sorted(DOCS.glob("*.md")) + [ROOT / "README.md", ROOT / "training" / "README.md"]
    for path in markdown:
        text = path.read_text(encoding="utf-8")
        for field in REQUIRED_METADATA:
            if f"**{field}:**" not in text:
                fail(errors, f"{path.relative_to(ROOT)}: missing metadata field {field!r}")


def validate_links(errors: list[str]) -> None:
    markdown = sorted(DOCS.glob("*.md")) + [ROOT / "README.md", ROOT / "training" / "README.md"]
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for source in markdown:
        for target in pattern.findall(source.read_text(encoding="utf-8")):
            target = target.strip().split(" ", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = target.split("#", 1)[0]
            if not relative:
                continue
            candidate = (source.parent / relative).resolve()
            if not candidate.exists():
                fail(errors, f"{source.relative_to(ROOT)}: broken relative link {target!r}")


def validate_markdown_contract_blocks(errors: list[str]) -> None:
    """Validate normative JSON examples embedded in Markdown.

    Illustrative snippets elsewhere in the documentation intentionally contain
    ellipses or type-union strings, so only fences explicitly marked with a
    ``contract=<schema-stem>`` info string are normative and machine-checked.
    """

    markdown = sorted(DOCS.glob("*.md")) + [ROOT / "README.md", ROOT / "training" / "README.md"]
    fence = re.compile(
        r"^[ \t]*```json[ \t]+contract=(?P<contract>[A-Za-z0-9._-]+)[ \t]*\n"
        r"(?P<body>.*?)^[ \t]*```[ \t]*$",
        re.MULTILINE | re.DOTALL,
    )
    for source in markdown:
        text = source.read_text(encoding="utf-8")
        for match in fence.finditer(text):
            contract = match.group("contract")
            schema_path = SCHEMAS / f"{contract}.schema.json"
            if not schema_path.exists():
                fail(errors, f"{source.relative_to(ROOT)}: no schema for embedded contract {contract!r}")
                continue
            try:
                value = json.loads(textwrap.dedent(match.group("body")).strip())
            except json.JSONDecodeError as exc:
                fail(errors, f"{source.relative_to(ROOT)}: invalid embedded {contract} JSON: {exc}")
                continue
            schema = read_json(schema_path, errors)
            if schema is None:
                continue
            for problem in validate_value(value, schema):
                fail(errors, f"{source.relative_to(ROOT)}: embedded {contract}: {problem}")


def validate_fixtures(errors: list[str]) -> None:
    schema_by_contract: dict[str, Path] = {}
    for path in sorted(SCHEMAS.glob("*.schema.json")):
        schema = read_json(path, errors)
        if schema is None:
            continue
        for field in ("$schema", "$id", "title", "type"):
            if field not in schema:
                fail(errors, f"{path.relative_to(ROOT)}: missing JSON Schema field {field!r}")
        title = schema.get("title")
        if isinstance(title, str):
            schema_by_contract[title] = path

    for path in sorted((EXAMPLES / "valid").glob("*.json")):
        value = read_json(path, errors)
        if value is None:
            continue
        contract = value.get("schema_version", "")
        schema_path = next((p for p in schema_by_contract.values() if p.stem.startswith(contract.split(".")[0])), None)
        if schema_path is None:
            # Filename mapping is the fallback for schemas whose title is not a simple prefix.
            schema_path = SCHEMAS / f"{path.stem}.schema.json"
        if not schema_path.exists():
            fail(errors, f"{path.relative_to(ROOT)}: no schema found for {contract!r}")
            continue
        schema = read_json(schema_path, errors)
        if schema is not None:
            problems = validate_value(value, schema)
            for problem in problems:
                fail(errors, f"{path.relative_to(ROOT)}: {problem}")

    for path in sorted((EXAMPLES / "invalid").glob("*.json")):
        value = read_json(path, errors)
        if value is None:
            continue
        schema_path = SCHEMAS / f"{path.stem.split('-')[0]}-v1.schema.json"
        # Explicit mapping handles descriptive invalid fixture names.
        mapping = {
            "capability-absent-as-occlusion.json": "capability-profile-v1.schema.json",
            "exercise-draft-approved.json": "exercise-recipe-v1.schema.json",
            "feature-width-378.json": "feature-schema-v1.schema.json",
            "workout-event-stale-decoder.json": "workout-event-v1.schema.json",
            "recommendation-empty-with-recipe.json": "eligible-recipe-set-v1.schema.json",
            "recommendation-catalog-mismatch.json": "workout-recommendation-v1.schema.json",
            "feedback-withdrawn-history.json": "workout-feedback-event-v1.schema.json",
            "dataset-missing-license.json": "dataset-manifest-v1.schema.json",
            "experiment-missing-provenance.json": "experiment-record-v1.schema.json",
            "artifact-mismatch.json": "model-artifact-manifest-v1.schema.json",
            "prediction-confidence-out-of-bounds.json": "movement-prediction-v1.schema.json",
            "recommendation-request-missing-profile.json": "recommendation-request-v1.schema.json",
        }
        schema_path = SCHEMAS / mapping.get(path.name, schema_path.name)
        schema = read_json(schema_path, errors) if schema_path.exists() else None
        if schema is None:
            fail(errors, f"{path.relative_to(ROOT)}: no schema found")
            continue
        schema_errors = validate_value(value, schema)
        policy_errors = validate_invalid_policy(path.name, value)
        if not schema_errors and not policy_errors:
            fail(errors, f"{path.relative_to(ROOT)}: expected fixture to be rejected")


def validate_invalid_policy(filename: str, value: dict[str, Any]) -> list[str]:
    """Apply cross-field rules that JSON Schema cannot express here."""

    problems: list[str] = []
    if filename == "workout-event-stale-decoder.json" and value.get("decoder_version") == "decoder-unknown":
        problems.append("decoder_version is not a declared decoder")
    if filename == "recommendation-empty-with-recipe.json" and value.get("empty") and value.get("recipe_ids"):
        problems.append("empty candidate set cannot contain recipe_ids")
    if filename == "recommendation-catalog-mismatch.json" and value.get("catalog_hash") == "sha256:old":
        problems.append("catalog hash is stale for the recommendation")
    if filename == "feedback-withdrawn-history.json" and value.get("consent_state") == "withdrawn" and value.get("event_type") == "completed":
        problems.append("withdrawn consent cannot record a completed history event")
    if filename == "experiment-missing-provenance.json" and not all(
        value.get(field) for field in ("git_commit", "config_hash", "dataset_manifest_ids", "artifact_path")
    ):
        problems.append("provenance fields are empty")
    if filename == "artifact-mismatch.json" and value.get("feature_schema_version") != "feature.v1":
        problems.append("movement-tcn-v1 requires feature.v1")
    return problems


def validate_current_facts(errors: list[str]) -> None:
    current = (DOCS / "current-state.md").read_text(encoding="utf-8")
    required_markers = ("283", "128", "96", "125", "zero", "no neural workout recommender")
    for marker in required_markers:
        if marker.lower() not in current.lower():
            fail(errors, f"docs/current-state.md: missing current-state marker {marker!r}")
    for directory, expected in (("artifacts/corrected-v1", "Complete benchmark"), ("artifacts/v2-quality", "Prepared data only"), ("artifacts/v2-quality-fixed", "Partial")):
        if not (ROOT / directory).exists():
            fail(errors, f"missing artifact directory required by current-state: {directory}")
        if expected.lower() not in current.lower():
            fail(errors, f"current-state does not state {directory} as {expected!r}")
    config = (ROOT / "training" / "configs" / "v1.yaml").read_text(encoding="utf-8")
    for marker in ("input_dim: 283", "window_frames: 128", "channels: 96"):
        if marker not in config:
            fail(errors, f"training/configs/v1.yaml: missing expected baseline marker {marker!r}")


def validate_registry_and_index(errors: list[str]) -> None:
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    for required in ("model-registry.md", "artifact-registry.md", "requirements-traceability.md", "schemas/", "historical"):
        if required not in index:
            fail(errors, f"docs/README.md: missing canonical navigation marker {required!r}")
    registry = (DOCS / "model-registry.md").read_text(encoding="utf-8")
    for model in ("movement-tcn-v1", "movement-gru-v1", "recommendation-ranker-v1", "af-mjepa-v1"):
        if model not in registry:
            fail(errors, f"docs/model-registry.md: missing model {model!r}")


def main() -> int:
    errors: list[str] = []
    validate_metadata(errors)
    validate_links(errors)
    validate_markdown_contract_blocks(errors)
    validate_fixtures(errors)
    validate_current_facts(errors)
    validate_registry_and_index(errors)
    if errors:
        print("documentation validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("documentation validation passed: metadata, links, fixtures, current facts, and registry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
