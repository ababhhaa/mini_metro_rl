import cv2
import numpy as np
import pickle
from main import capture_screen, isolate_white_shapes, mask_ui

# Global state for click callback
selected_contour = None
contours_list    = []
hierarchy        = None

def find_contour_at(x, y):
    """Return the outermost contour whose bounding box contains (x, y)."""
    for i, c in enumerate(contours_list):
        if hierarchy[0][i][2] == -1:  # skip non-stations (no child)
            continue
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cx <= x <= cx + cw and cy <= y <= cy + ch:
            return c
    return None

def on_click(event, x, y, flags, param):
    global selected_contour
    if event == cv2.EVENT_LBUTTONDOWN:
        c = find_contour_at(x, y)
        if c is not None:
            selected_contour = c
            print(f"Contour selected at ({x}, {y}) — press a key to label it")

def main():
    global contours_list, hierarchy, selected_contour

    templates = {}  # Shape name string → contour

    frame = capture_screen()
    frame = mask_ui(frame)
    white_only = isolate_white_shapes(frame)
    gray = cv2.cvtColor(white_only, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours_list, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # Draw all station contours on debug frame
    debug = frame.copy()
    for i, c in enumerate(contours_list):
        if hierarchy[0][i][2] != -1:  # stations only
            cv2.drawContours(debug, [c], 0, (0, 255, 0), 2)
            cx, cy, cw, ch = cv2.boundingRect(c)
            cv2.putText(debug, str(i), (cx, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    cv2.imshow("Template Extractor — click a station, then press its label key", debug)
    cv2.setMouseCallback("Template Extractor — click a station, then press its label key", on_click)

    # Key → shape name mapping
    key_map = {
        ord('c'): 'circle',
        ord('t'): 'triangle',
        ord('s'): 'square',
        ord('p'): 'pentagon',
        ord('x'): 'cross',
        ord('w'): 'wedge',
        ord('g'): 'gem',
        ord('d'): 'diamond',
        ord('o'): 'oval',
        ord('*'): 'star',    # press shift+8
    }

    print("Controls: click a green contour, then press the label key")
    print("  c=circle  t=triangle  s=square  p=pentagon")
    print("  x=cross   w=wedge     g=gem      d=diamond  o=oval  *=star")
    print("  q=quit and save")

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break

        if key in key_map and selected_contour is not None:
            name = key_map[key]
            templates[name] = selected_contour
            print(f"Saved template: {name}")

            # Draw confirmation on debug frame
            cx, cy, cw, ch = cv2.boundingRect(selected_contour)
            cv2.putText(debug, name, (cx, cy - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            cv2.imshow("Template Extractor — click a station, then press its label key", debug)
            selected_contour = None

    cv2.destroyAllWindows()

    # Save templates to disk
    with open("templates.pkl", "wb") as f:
        pickle.dump(templates, f)
    print(f"Saved {len(templates)} templates to templates.pkl")
    print("Shapes saved:", list(templates.keys()))

if __name__ == "__main__":
    main()