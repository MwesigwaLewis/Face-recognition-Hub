import cv2


def draw_corner_box(frame, x1, y1, x2, y2,
                    color=(120, 200, 255),
                    thickness=1,
                    length=25):

    # Top-left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)

    # Top-right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)

    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)

    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

    # Mid-line
    cv2.line(frame, (x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2), color, thickness)


def draw_name_plate(frame, name, confidence, x, y):

    text = f"{name}  {confidence*100:.1f}%"

    cv2.rectangle(
        frame,
        (x,y+20),
        (x+220,y+70),
        (120,120,120),
        -1
    )


    cv2.putText(
        frame,
        text,
        (x+10,y+55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )