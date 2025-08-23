from ultralytics import YOLO
import cv2

RECYCLABLE = ['cardboard_box','can','plastic_bottle_cap','plastic_bottle','reuseable_paper']
NON_RECYCLABLE = ['plastic_bag','scrap_paper','stick','plastic_cup','snack_bag','plastic_box','straw',
                  'plastic_cup_lid','scrap_plastic','cardboard_bowl','plastic_cultery']
HAZARDOUS = ['battery','chemical_spray_can','chemical_plastic_bottle','chemical_plastic_gallon',
             'light_bulb','paint_bucket']

CATEGORY_WEIGHTS = {
    "recyclable": 0.5,
    "non_recyclable": 0.7,
    "hazardous": 1.0
}

def get_category(label):
    if label in RECYCLABLE:
        return "recyclable"
    elif label in NON_RECYCLABLE:
        return "non_recyclable"
    elif label in HAZARDOUS:
        return "hazardous"
    else:
        return "unknown"

def analyze_waste(weights_path, image_path):
    """
    Run YOLOv8 inference on a waste image and compute overall confidence and severity.
    """
    # Load model
    model = YOLO(weights_path)

    # Run inference
    results = model(image_path)

    # Load image to calculate areas
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    img_area = H * W

    confidences = []
    weighted_area = 0

    for result in results:
        names = result.names  # class id -> label mapping
        for box in result.boxes:
            cls = int(box.cls[0])
            label = names[cls]
            conf = float(box.conf[0])
            confidences.append(conf)

            # bounding box area
            x1, y1, x2, y2 = box.xyxy[0]
            box_area = (x2 - x1) * (y2 - y1)

            # category weight
            category = get_category(label)
            weight = CATEGORY_WEIGHTS.get(category, 0.5)

            weighted_area += box_area * weight

    if len(confidences) == 0:
        return 0.0, 0.0

    # confidence = max detection confidence
    overall_conf = max(confidences)

    # severity = weighted area / image area (clamped 0–1)
    severity = min(weighted_area / img_area, 1.0)

    return overall_conf, severity

conf, sev = analyze_waste("models/trash_new.pt", "t.jpeg")
print(f"Overall Confidence: {conf:.3f}")
print(f"Overall Severity: {sev:.3f}")
