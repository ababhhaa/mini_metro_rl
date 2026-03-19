import mss
import numpy as np
import cv2
from enum import Enum
import pickle

class Shape(Enum):
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    SQUARE = "square"
    PENTAGON = "pentagon"
    STAR = "star"
    CROSS = "cross"
    WEDGE = "wedge"
    GEM = "gem"
    DIAMOND = "diamond"
    OVAL = "oval"
    NONE = "none"


def capture_screen() -> np.ndarray:
    """Capture the full screen as a BGR numpy array."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def isolate_white_shapes(frame: np.ndarray) -> np.ndarray:
    # Convert to HSV — easier to define "white" here than in BGR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_ui(frame)
    # White = low saturation, high value (brightness)
    lower_white = np.array([0,   0,   200])  # H, S, V
    upper_white = np.array([180, 40,  255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply mask — non-white pixels become black
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result


def mask_ui(frame: np.ndarray) -> np.ndarray:

    # Top-left back arrow
    frame[0:84, 0:99] = 0

    # Top-right score/clock/timer
    frame[0:90, 1619:] = 0 # top rect
    frame[0:237, 1824:] = 0 # vert bottom rect
    return frame


def load_templates(path="templates.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

def classify_contour(contour, templates: dict) -> Shape:
    # Step 1 — matchShapes against all templates
    scores = {
        name: cv2.matchShapes(contour, template, cv2.CONTOURS_MATCH_I1, 0)
        for name, template in templates.items()
    }
    best = min(scores, key=scores.get)

    # Step 2 — refinement for ambiguous pairs
    best = refine(contour, best, scores)

    return Shape(best)


def refine(contour, best: str, scores: dict) -> str:
    x, y, w, h = cv2.boundingRect(contour)
    aspect = max(w, h) / min(w, h)

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    area      = cv2.contourArea(contour)
    solidity  = area / hull_area

    approx   = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    vertices = len(approx)

    # square vs diamond
    if best in ('square', 'diamond'):
        return 'diamond' if aspect > 1.2 else 'square'

    # circle vs oval — lower threshold, oval is only slightly elongated
    if best in ('circle', 'oval'):
        return 'oval' if aspect > 1.15 else 'circle'

    # star vs cross vs triangle — use solidity + vertices
    if best in ('star', 'cross', 'triangle'):
        if vertices == 3:    return 'triangle'   # triangles have exactly 3 vertices
        if solidity < 0.55:  return 'star'       # very non-convex = star
        return 'cross'                            # blockier non-convex = cross

    # gem vs diamond vs square
    if best in ('gem', 'diamond', 'square') and vertices == 4:
        if aspect < 1.2:  return 'square'
        if aspect < 1.8:  return 'gem'
        return 'diamond'

    return best

def main():
    frame = capture_screen()
    frame = mask_ui(frame)
    white_only = isolate_white_shapes(frame)
    gray = cv2.cvtColor(white_only, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours_list, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    debug = frame.copy()  # use the original color frame so you can see context

    stations = [c for i, c in enumerate(contours_list) if hierarchy[0][i][2] != -1]  # has child all outer contours
    passengers = [c for i, c in enumerate(contours_list) if (hierarchy[0][i][2] == -1 and hierarchy[0][i][3] == -1)]  # has no child and no parent
    # Draw ALL contours

    templates = load_templates()

    for i, c in enumerate(stations):
        shape = classify_contour(c, templates)

        cx, cy, cw, ch = cv2.boundingRect(c)
        cv2.drawContours(debug, [c], 0, (0, 255, 0), 2)
        cv2.putText(debug, shape.value+"Station", (cx, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    for i, c in enumerate(passengers):
        shape = classify_contour(c, templates)

        cx, cy, cw, ch = cv2.boundingRect(c)
        cv2.drawContours(debug, [c], 0, (0, 255, 0), 2)
        cv2.putText(debug, shape.value, (cx, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

    cv2.imwrite("contours.png", debug)
    cv2.imshow("contours", debug)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()