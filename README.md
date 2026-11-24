# FlockCam

Real-time facial recognition for a camera feed. FlockCam enrolls **known faces**. On a live
video stream it then labels each face it sees as a known person (green box) or an
**unknown** face (red box). Every new sighting goes to a JSON-lines event log, and
unknown faces are saved as snapshots for review.

```
webcam / RTSP / video file
        │
        ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌───────────────┐
 │ YuNet        │──▶│ SFace        │──▶│ FaceDatabase │──▶│ EventRecorder │──▶ events.jsonl
 │ detect faces │   │ 128-d embed  │   │ cosine match │   │ + cooldown    │──▶ snapshots/*.jpg
 └──────────────┘   └──────────────┘   └──────────────┘   └───────────────┘
                                              │
                                     UnknownRegistry  ("Unknown #3" stays #3)
                                              │
                                              ▼
                                       overlay → window
```

## Quick start (Windows, PowerShell)

```powershell
git clone https://github.com/<you>/flockcam.git
cd flockcam
py -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -e ".[dev]"

flockcam download-models              # ~40 MB, once
pytest                                # all tests, incl. real-model smoke tests

# Enroll yourself: 5 webcam samples (turn your head a little between each)
flockcam enroll "David" --camera --samples 5
# ...or from photos (files or a folder)
flockcam enroll "Alice" photos\alice\

flockcam list
flockcam run                          # q / Esc to quit, e to enroll the largest face in view
```

macOS/Linux: same commands, but activate with `source .venv/bin/activate`.

## Commands

| Command | What it does |
|---|---|
| `flockcam download-models` | Fetches the OpenCV Zoo YuNet detector and SFace recognizer (ONNX) into `models/` |
| `flockcam enroll NAME [IMAGES...] [--camera --samples N]` | Adds one or more face samples for a person |
| `flockcam list` / `flockcam remove NAME` | Manage the database |
| `flockcam run [--source 0\|rtsp://...\|file.mp4] [--cooldown 10] [--every 2] [--no-window]` | Live recognition |
| Global: `--threshold 0.363`, `--data-dir data`, `--models-dir models` | Tuning and paths |

Sample event (`data/events.jsonl`):
```json
{"timestamp": "2026-11-24T00:41:07.123+00:00", "label": "Unknown #1", "known": false,
 "similarity": 0.1204, "confidence": 0.93, "box": [212, 98, 141, 177],
 "snapshot": "data/snapshots/20261124T004107_123000_Unknown_1.jpg"}
```

## Project layout

```
src/flockcam/
  types.py      value objects: BoundingBox, Detection, MatchResult, RecognizedFace
  engines.py    FaceDetector / FaceEmbedder protocols + YuNet / SFace implementations
  database.py   embedding store, vectorised cosine matching, atomic save/load
  pipeline.py   frame -> recognised faces; UnknownRegistry for stable stranger IDs
  events.py     event log + snapshots, per-identity cooldown
  overlay.py    drawing
  camera.py     capture (DirectShow on Windows) and the live loop
  models.py     model download
  cli.py        argparse entry point
tests/          43 unit/integration tests + real-model smoke tests
```

## Design decisions

- **Two-stage detect → embed → nearest neighbour.** Recognition is an open-set problem: the
  system has to say "none of the above". Comparing embeddings against a similarity threshold
  handles that, and a new person can be added without retraining anything.
- **OpenCV YuNet + SFace.** Only `opencv-python` is needed. There is no dlib/CMake
  build, so installs are painless on Windows. Both models run in real time on a laptop CPU.
  The threshold of 0.363 is OpenCV's published cosine threshold for SFace.
- **Dependency inversion.** The pipeline depends on the `FaceDetector`/`FaceEmbedder`
  protocols, not on OpenCV. The tests inject fakes where a "face" is a coloured square and
  its colour is the embedding. That makes the matching, tracking, event, CLI and live-loop
  logic deterministic to test with no camera and no model files.
- **Several samples per person.** A probe matches if it is close to *any* sample. That makes
  recognition hold up better across pose and lighting.
- **Stable IDs for unknown faces plus a cooldown.** Without them, one stranger in front of the
  camera would produce a new alert and snapshot on every frame.
- **Operational details.** Database writes are atomic (temp file then rename). Faces below a
  minimum size are ignored. `--every N` trades latency for CPU. `--no-window` runs headless for
  a server or RTSP deployment.

## Limitations and next steps

- No liveness or anti-spoofing check: a printed photo can match.
- Matching is a linear scan, which is fine up to ~100k embeddings. Beyond that, use
  FAISS/pgvector.
- Tracking is by embedding only. A spatial tracker (IoU/SORT) would cut embedding work
  between frames.
- Threshold tuning: calibrate on your own camera with a labelled set (an ROC curve) rather than
  relying on the published default.
- Alerting hooks (webhook/Slack) on unknown sightings would plug into `EventRecorder`.

## Privacy

Face embeddings are biometric data, and many jurisdictions (GDPR, Illinois BIPA, etc.)
require consent and limits on how long it is kept. `data/` and `models/` are git-ignored, so
no biometric data or weights ever reach the repo. Only enroll people who have agreed to it.

## License

Code: MIT. Models: OpenCV Zoo (Apache-2.0).
