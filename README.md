## CrashDetect
CrashDetect is a realtime car crash detection system designed to reduce post crash response time and save lives.

### Getting Started
After cloning the repository create a python3 virtual enviornment using the command `python -m venv venv`. Next, activate this virtual enviornment by running the command `source venv/bin/activate`.

### CrashDetect Architecture
Crashdetect is an end-to-end car crash detection system for surveillance cameras. The first step is to use computer vision techniques to zoom into regions with car movement in the video frame. This dramatically reduces computation and improves model intelligence. This code is available in the [car identification](#car-identification) section. Next the smaller video segments are analysed by a neural network. This neural network predicts whether the small video has a car crash or if it does not. This output could be used in the future to notify emergency services. Read more about the neural network under the [video analysis](#video-analysis) section.

### Car Identification
CrashDetect first finds regions in the video stream with high movement, indicating areas with potential car movement. You can try this by running the car_identification.py file using the command `python3 car_identification.py`.

### Video Analysis


