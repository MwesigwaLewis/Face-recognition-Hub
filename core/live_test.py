import cv2

from .camera import Camera
from .recognizer import recognize
from .drawing import draw_corner_box, draw_name_plate


camera = Camera()

frame_count = 0
last_results = []


while True:

    frame = camera.get_frame()

    if frame is None:
        break

    frame_count += 1

    # Run recognition every 5th frame
    if frame_count % 5 == 0:
        last_results = recognize(frame)

    # Draw all detected faces
    for result in last_results:

        x1, y1, x2, y2 = map(int, result["bbox"])

        draw_corner_box(
            frame,
            x1,
            y1,
            x2,
            y2
        )

        cv2.putText(
            frame,
            result["name"],
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{result['score']:.2f}",
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    '''
        draw_name_plate(
        frame,
        result["name"],
        result["score"],
        x1,
        y2
    )
    '''

    cv2.imshow(
        "#Lewiscrypt",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("b"):
        break


camera.release()
cv2.destroyAllWindows()