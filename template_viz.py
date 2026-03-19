import pickle
import cv2
import numpy as np

def visualize_templates(path="templates.pkl"):
    with open(path, "rb") as f:
        templates = pickle.load(f)

    cell_size = 150
    cols = 5
    rows = -(-len(templates) // cols)  # ceiling division

    canvas = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

    for idx, (name, contour) in enumerate(templates.items()):
        row = idx // cols
        col = idx % cols

        # Create blank cell
        cell = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)

        # Normalize contour to fit the cell with padding
        padding = 20
        x, y, w, h = cv2.boundingRect(contour)
        scale = (cell_size - padding * 2) / max(w, h)

        # Shift + scale contour to center of cell
        normalized = contour.copy().astype(np.float32)
        normalized[:, 0, 0] = (normalized[:, 0, 0] - x) * scale + padding + (cell_size - padding*2 - w*scale) / 2
        normalized[:, 0, 1] = (normalized[:, 0, 1] - y) * scale + padding + (cell_size - padding*2 - h*scale) / 2
        normalized = normalized.astype(np.int32)

        cv2.drawContours(cell, [normalized], 0, (0, 255, 0), 2)

        # Label
        cv2.putText(cell, name, (5, cell_size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Place cell in canvas
        canvas[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size] = cell

    cv2.imwrite("templates_viz.png", canvas)
    cv2.imshow("Templates", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_templates()