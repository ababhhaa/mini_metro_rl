import mss
import numpy as np
import cv2

from enum import Enum

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

def classify_contours(contour) -> Shape:
    #TODO
    return Shape.NONE

def main():
    frame = capture_screen()
    white_only = isolate_white_shapes(frame)
    gray = cv2.cvtColor(white_only, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours_list, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    debug = frame.copy()  # use the original color frame so you can see context

    # Draw ALL contours
    cv2.drawContours(debug, contours_list, -1, (139, 255, 50), 2)
    cv2.imshow("all contours", debug)
    cv2.imwrite("contours.png", debug)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()