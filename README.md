# Screen Time Project

This project aims to calculate the screen time of each character in a video using face detection and other techniques. The program detects faces in videos, then calculates each person's total time in the video.

## Implementation

The program is written in Python and uses the OpenCV library for face detection. The face detection algorithm is based on the Haar Cascade Classifier and uses a pre-trained model to detect faces in the video frames. The program then tracks each face and calculates the total time each person is on screen.

## Data Sources

The project uses video files as input. The videos are in MP4 format and were obtained from [source]. The video data is preprocessed to extract frames for face detection.

## Usage

To run the program, first install the required dependencies by running `pip install -r requirements.txt`. Then, run the command `python main.py [video file]` to calculate the screen time for the characters in the video. The program outputs the results in a CSV file.

## Results

The program successfully calculates the screen time for each character in the video. The results are visualized in a pool chart that shows which character is seen most in the episodes and seasons.

## Repository Structure

- `README.md`: project overview and usage instructions
- `main.py`: main program file
- `utils.py`: utility functions
- `data/`: directory containing input video files
- `results/`: directory containing output CSV files
