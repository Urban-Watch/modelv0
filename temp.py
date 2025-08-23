# -*- coding: utf-8 -*-
\"\"\"run_models_and_describe.py

Modified behavior: for each detector (pothole, graffitti, trash) the script will:
 - run the detector to get severity + confidence
 - call the specified Gemini model separately with the image and that single detector's output
 - collect per-detector textual descriptions from Gemini

Usage:
    python run_models_and_describe.py --image /path/to/image.jpg
        [--pothole pothole_agent.py] [--graffiti graffitti_agent.py] [--trash trash_agent.py]
        [--gemini_model gemini-flash-1.5-latest] [--api_key_env GEMINI_API_KEY]

Outputs printed on stdout include per-model raw outputs, per-model Gemini descriptions,
and an optional aggregated summary (same heuristic as before).

Requirements:
    pip install google-genai Pillow
\"\"\"

import argparse
import importlib.util
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict

from PIL import Image

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None

# --- dynamic loader & predictor extraction (same robust behavior) ---

def load_module_from_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f\"Module file not found: {path}\")
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f\"Could not load module from {path}\")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

def make_predictor(module_or_path: Any) -> Callable[[Image.Image], Dict[str, Any]]:
    if isinstance(module_or_path, str):
        module = load_module_from_path(module_or_path)
    else:
        module = module_or_path

    candidate_names = [\"predict\", \"run\", \"infer\", \"classify\", \"detect\"]
    for name in candidate_names:
        if hasattr(module, name) and callable(getattr(module, name)):
            fn = getattr(module, name)
            return lambda img, _fn=fn: _fn(img)

    for attr in dir(module):
        if attr.lower() in (\"model\", \"agent\", \"predictor\", \"classifier\"):
            cls = getattr(module, attr)
            try:
                inst = cls()
                if hasattr(inst, \"predict\") and callable(getattr(inst, \"predict\")):
                    return lambda img, _inst=inst: _inst.predict(img)
            except Exception:
                if hasattr(cls, \"predict\") and callable(getattr(cls, \"predict\")):
                    return lambda img, _cls=cls: _cls.predict(img)

    if callable(module):
        return lambda img, _m=module: _m(img)

    raise AttributeError(
        f\"No suitable predictor callable found in module {getattr(module, '__file__', str(module))}. \"
        \"Expected a function named predict/run/infer or a Model/Agent class with .predict(image).\"
    )

# --- severity normalization ---

def severity_to_numeric(s):
    if s is None:
        return 0.0
    try:
        v = float(s)
        if v > 1.0:
            if v <= 100:
                return max(0.0, min(1.0, v / 100.0))
            return max(0.0, min(1.0, v / 1000.0))
        return max(0.0, min(1.0, v))
    except Exception:
        pass
    ss = str(s).strip().lower()
    mapping = {
        \"none\": 0.0,
        \"no\": 0.0,
        \"low\": 0.2,
        \"minor\": 0.2,
        \"med\": 0.5,
        \"medium\": 0.5,
        \"moderate\": 0.5,
        \"high\": 0.8,
        \"severe\": 1.0,
        \"critical\": 1.0,
    }
    return mapping.get(ss, 0.5)

def numeric_to_severity_label(v: float) -> str:
    if v < 0.25:
        return \"low\"
    if v < 0.6:
        return \"medium\"
    if v < 0.85:
        return \"high\"
    return \"critical\"

# --- orchestration ---

def predict_all(image_path: str, agents: Dict[str, str]):
    img = Image.open(image_path).convert(\"RGB\")
    predictors = {}
    for k, path in agents.items():
        try:
            predictors[k] = make_predictor(path)
        except Exception as e:
            raise RuntimeError(f\"Error preparing predictor for {k}: {e}\\n{traceback.format_exc()}\")
    results = {}
    with ThreadPoolExecutor(max_workers=len(predictors)) as ex:
        futures = {ex.submit(pred, img): name for name, pred in predictors.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                raw = fut.result()
            except Exception as e:
                results[name] = {
                    \"error\": str(e),
                    \"traceback\": traceback.format_exc()
                }
                continue
            severity = raw.get(\"severity\") if isinstance(raw, dict) else None
            confidence = raw.get(\"confidence\") if isinstance(raw, dict) else None
            try:
                severity_score = severity_to_numeric(severity)
            except Exception:
                severity_score = 0.0
            try:
                confidence_score = float(confidence) if confidence is not None else 0.0
            except Exception:
                try:
                    confidence_score = float(confidence) / 100.0
                except Exception:
                    confidence_score = 0.0
            results[name] = {
                \"raw\": raw,
                \"severity\": severity,
                \"severity_score\": severity_score,
                \"confidence\": confidence_score
            }
    return results

def aggregate_results(results: Dict[str, Any]):
    severity_scores = [v.get(\"severity_score\", 0.0) for v in results.values() if isinstance(v, dict)]
    confidences = [v.get(\"confidence\", 0.0) for v in results.values() if isinstance(v, dict)]
    overall_severity_score = max(severity_scores) if severity_scores else 0.0
    overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        \"overall_severity_score\": overall_severity_score,
        \"overall_severity_label\": numeric_to_severity_label(overall_severity_score),
        \"overall_confidence\": overall_confidence
    }

# --- Gemini interaction: call model for a SINGLE detector result ---

def call_gemini_for_detector(image_path: str, detector_name: str, detector_result: Dict[str, Any],
                             model_name: str = \"gemini-flash-1.5-latest\", api_key_env: str = \"GEMINI_API_KEY\",
                             max_tokens: int = 512):
    if genai is None or genai_types is None:
        raise RuntimeError(\"google-genai SDK is not installed or couldn't be imported. Install with: pip install google-genai\") 
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f\"Gemini API key not found in environment variable {api_key_env}.\")
    client = genai.Client(api_key=api_key)

    # Build a focused prompt for this detector only
    prompt_lines = [
        f\"You are an assistant that inspects an image and the output of a single automated detector: {detector_name}.\",
        \"Produce a concise (2-4 short paragraphs) human-readable description of what the detector likely detected in the image.\",
        \"Emphasize:\",
        \"- What the image shows (objects, context) as relevant to this detector.\",
        \"- The detector's reported severity and confidence, and what they mean in plain language.\",
        \"- Whether immediate action is recommended based on the detector's severity and confidence.\",
        \"FORMAT: Start with a one-sentence summary, then one short paragraph explaining the detector reading, then a recommended action line.\",
        \"\",
        \"Detector output (severity_label / severity_score[0..1] / confidence[0..1]):\",
    ]
    if isinstance(detector_result, dict) and \"error\" not in detector_result:
        prompt_lines.append(f\"- {detector_name}: {detector_result.get('severity')} / {detector_result.get('severity_score'):.3f} / {detector_result.get('confidence'):.3f}\")
    else:
        prompt_lines.append(f\"- {detector_name}: ERROR - {detector_result.get('error', 'unknown')}\")

    prompt_lines.append(\"\\nNow describe the image (briefly) and provide the requested summary and recommended action for this detector.\")

    prompt_text = \"\\n\".join(prompt_lines)

    # Read image bytes
    mime_type = \"image/jpeg\"
    ext = os.path.splitext(image_path)[1].lower()
    if ext in (\".png\", \".webp\", \".heic\", \".heif\"):
        if ext == \".png\": mime_type = \"image/png\"
        elif ext == \".webp\": mime_type = \"image/webp\"
        elif ext == \".heic\": mime_type = \"image/heic\"
        elif ext == \".heif\": mime_type = \"image/heif\"

    with open(image_path, \"rb\") as f:
        image_bytes = f.read()

    image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    contents = [image_part, prompt_text]

    response = client.models.generate_content(model=model_name, contents=contents)
    text = None
    try:
        text = response.text
    except Exception:
        try:
            text = response.text()
        except Exception:
            text = str(response)
    return text

# --- CLI entrypoint ---

def main():
    ap = argparse.ArgumentParser(description=\"Run detectors and call Gemini separately for each detector's output.\")
    ap.add_argument(\"--image\", required=True, help=\"Path to input image\")
    ap.add_argument(\"--pothole\", default=\"pothole_agent.py\", help=\"Path to pothole agent module (.py)\")
    ap.add_argument(\"--graffiti\", default=\"graffitti_agent.py\", help=\"Path to graffitti agent module (.py)\")
    ap.add_argument(\"--trash\", default=\"trash_agent.py\", help=\"Path to trash agent module (.py)\")
    ap.add_argument(\"--gemini_model\", default=\"gemini-flash-1.5-latest\", help=\"Gemini model name to call (you can override)\")
    ap.add_argument(\"--api_key_env\", default=\"GEMINI_API_KEY\", help=\"Env var name holding Gemini API key\")
    ap.add_argument(\"--no_gemini\", action=\"store_true\", help=\"If set, skip calling the LLM and just print the aggregated JSON\")
    args = ap.parse_args()

    agents = {
        \"pothole_agent\": args.pothole,
        \"graffitti_agent\": args.graffiti,
        \"trash_agent\": args.trash
    }

    print(\"Loading and running detectors in parallel...\")
    per_model = predict_all(args.image, agents)

    print(\"\\nPer-model outputs:\")
    for k, v in per_model.items():
        print(f\"== {k} ==\")
        print(v)
        print(\"\")

    aggregated = aggregate_results(per_model)
    print(\"Aggregated:\", aggregated)

    if args.no_gemini:
        print(\"\\n(no gemini) Done.\")
        return

    # Call Gemini separately for each detector (sequential to avoid rate issues)
    print(\"\\nCalling Gemini separately for each detector...\") 
    per_model_descriptions = {}
    for name, result in per_model.items():
        print(f\"\\n--- Calling Gemini for {name} ---\")
        try:
            desc = call_gemini_for_detector(args.image, name, result,
                                            model_name=args.gemini_model, api_key_env=args.api_key_env)
            per_model_descriptions[name] = desc
            print(\"Description:\\n\", desc)
        except Exception as e:
            per_model_descriptions[name] = f\"ERROR: {e}\\n{traceback.format_exc()}\"
            print(\"Error calling Gemini for\", name, e)
            print(traceback.format_exc())

    # Optionally print an aggregated recommendation summary using the previous heuristic
    print(\"\\n--- Aggregated summary (heuristic) ---\\n\", aggregated)

    print(\"\\nDone.\")


if __name__ == \"__main__\":
    main()
