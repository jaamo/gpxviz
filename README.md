# GPX Visualizer

A Python-based tool for visualizing GPX track files by rendering animated, glowing lines. This project converts GPX data into smooth visual animations and exports the result as a video or image sequence.

Created with Python 3.13.2

## Features

- Supports multiple GPX files
- Renders smooth glowing line animations
- Outputs high-quality image sequences
- Customizable window orientation (horizontal or vertical)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jaamo/gpxviz.git
   cd gpxviz
   ```

2. Install dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

## Usage

1. Save your `.gpx` files in the `tracks/` folder.
2. Run the renderer:
   ```bash
   python3 main.py
   ```
3. Check the output in the `output/` folder.

## Convert images to a video

```bash
ffmpeg -framerate 30 -pattern_type glob -i 'output/*.jpg' -c:v libx264 -pix_fmt yuv420p output.mp4
```

## Bounding Box Finder

Use these URLs to find the bounding box for your region:

- **Horizontal Layout:**
  [bboxfinder.com - Horizontal](http://bboxfinder.com/#60.100646,24.301758,60.380124,25.353355)

- **Vertical Layout:**
  [bboxfinder.com - Vertical](http://bboxfinder.com/#60.081968,24.551010,60.487032,25.057068)
