# Face Recognition Hub (PySide6)

A dark, HUD-style desktop UI for a face recognition app — camera feed with
bounding box / landmark overlay, identity card, sidebar with system status
and recognized people, top bar clock, bottom nav.

## Install

```bash
pip install PySide6 opencv-python numpy
```

## Run

```bash
python main.py
```

It launches with a `DemoEngine` that fakes a detection so you can see the
UI working immediately (webcam required for the video feed itself).

## Plugging in your existing recognition code

Everything goes through **one file**: `recognition_interface.py`.

1. Subclass `RecognitionEngine`:

```python
from recognition_interface import RecognitionEngine, FaceResult

class MyEngine(RecognitionEngine):
    def __init__(self):
        self.known = load_known_faces()   # your existing setup

    def process_frame(self, frame):
        # frame is a BGR numpy array from OpenCV
        boxes, encodings = my_detector.detect(frame)
        results = []
        for box, enc in zip(boxes, encodings):
            name, score = my_matcher.match(enc, self.known)
            results.append(FaceResult(
                name=name or "UNKNOWN",
                confidence=score,       # 0.0 - 1.0
                box=box,                # (x, y, w, h) in frame pixels
                landmarks=[],           # optional [(x, y), ...]
                person_id=None,         # optional int
            ))
        return results
```

2. In `main.py`, swap the engine:

```python
from my_engine import MyEngine
engine = MyEngine()
```

That's it — the video overlay, identity card, sidebar list, and match bars
all update automatically from whatever `FaceResult` list you return.

## Files

- `main.py` — app entry point, camera loop, wiring
- `recognition_interface.py` — the interface you implement (only file to edit)
- `video_panel.py` — video display + HUD overlay painting (QPainter)
- `sidebar.py` — system status + recognized people list
- `chrome_bars.py` — top bar (clock) and bottom nav
- `style.qss` — dark green theme

## Notes

- Bottom nav buttons currently just toggle visual "active" state — wire
  `BottomNav.buttons["enroll"].clicked` etc. to a `QStackedWidget` if you
  want separate Enroll/People/Settings/Logs pages.
- `VideoPanel` letterboxes the camera frame to fit any window size, so
  overlay coordinates stay correctly aligned when resizing.
