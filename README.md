# PrimitiveQA

**The quality layer for physical AI data.**

PrimitiveQA takes any hand manipulation video, decomposes it into universal skill primitives (reach, grasp, lift, transport, place, retract), and scores the quality of each primitive. The output tells you what's good, what's bad, and what's missing in your training data.

**Why it matters:** Raw manipulation video is worth ~$1/hr to robotics labs. Labeled teleoperation data is worth $100–200/hr. The gap exists because raw video has no structured action information. PrimitiveQA bridges that by extracting structured primitives and quality scores from any video source.

---

## What's in this repo

```
primitiveqa/
├── app.py                  # Gradio web UI — the demo entrypoint
├── pipeline.py             # Orchestrator: wires all modules together
├── core/
│   └── types.py            # Shared data structures (Trajectory, Segment, QualityScore, etc.)
├── sources/
│   └── phone_video.py      # MediaPipe Hand Landmarker — extracts 21-point skeleton from video
├── segmentation/
│   └── segmenter.py        # Decomposes trajectory into primitives using adaptive heuristics
├── scoring/
│   └── scorer.py           # Computes 4 quality metrics per segment + composite score
├── evaluation/
│   └── vlm.py              # Claude vision call — task success + description (optional)
├── visualization/
│   └── charts.py           # Plotly radar chart, primitive timeline, per-segment bars
├── models/                 # MediaPipe hand landmarker model (downloaded on setup, gitignored)
├── examples/               # Your demo video clips (gitignored)
└── output/                 # JSON export outputs (gitignored)
```

---

## Setup

**Requirements:** Python 3.10+, ~8GB RAM, no GPU needed.

```bash
# 1. Clone the repo
git clone https://github.com/divyav9701/primitiveqa.git
cd primitiveqa

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the MediaPipe hand landmarker model
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

**Optional — Claude vision evaluation:**
```bash
# Create a .env file with your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```
The app works without this — the Claude panel will show a placeholder. Cost is ~$0.02/video when enabled.

---

## Running the app

```bash
source .venv/bin/activate
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

1. Upload a short hand manipulation video (10–30 seconds, phone camera works great)
2. Paste your Anthropic API key if you want the Claude evaluation (optional)
3. Click **Analyze**
4. View the skeleton overlay, primitive timeline, quality radar, and per-segment scores
5. Download the JSON export

---

## Quality metrics

Each primitive segment is scored on four dimensions:

| Metric | What it measures |
|---|---|
| **Smoothness** | Low jerk (3rd derivative of wrist position) — jerky motion scores low |
| **Path efficiency** | Straight-line vs. actual path length — wandering scores low |
| **Decisiveness** | Low velocity variance — hesitation and speed changes score low |
| **Detection confidence** | Mean MediaPipe hand confidence — occlusion and blur score low |

These combine into a **composite score** (0–1). A clean, deliberate pick-and-place should score above 0.75.

---

## Tips for good demo clips

- **Camera angle:** Phone propped up at roughly 45°, hand visible from above-ish
- **Lighting:** Good room light, avoid backlighting
- **Task:** Pick up an object (cup, pen, bottle), move it across the desk, set it down
- **Duration:** 10–15 seconds is ideal — long enough for the full reach→grasp→lift→transport→place→retract sequence
- **Variation:** Record 3 clips — one deliberate and clean, one fast/sloppy, one where your hand goes partially out of frame — to show score contrast in the demo

---

## Tech stack

| Component | Tool |
|---|---|
| Web UI | Gradio 4.x |
| Hand tracking | MediaPipe Hand Landmarker (Tasks API) |
| Video I/O | OpenCV |
| Quality scoring | NumPy + SciPy |
| Task evaluation | Anthropic Claude API (`claude-sonnet-4-6`) |
| Visualization | Plotly |
