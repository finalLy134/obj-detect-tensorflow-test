# Object Detection with TensorFlow and tkinter

This project demonstrates real-time object detection using a pre-trained TensorFlow model integrated into a tkinter-based graphical user interface (GUI).

## Features

- **Object Detection** 📷: Detects objects in images using a pre-trained TensorFlow model.
- **GUI Interface** 🖼️: Simple tkinter GUI for selecting and analyzing images.
- **Bounding Box Visualization** 🎯: Draws bounding boxes and labels around detected objects.

## Requirements

- Python 3.x 🐍
- TensorFlow 2.x 🧠
- tkinter 🖥️
- OpenCV 📸
- Pillow (PIL) 🖼️

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd ObjectDetection-TensorFlow-tkinter
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained model (`saved_model`) and label map (`label_map.pbtxt`) from TensorFlow's model zoo or provide your own.

## Usage

1. Run the application:

   ```bash
   python main.py
   ```

2. Click on "Choose Image" to browse and select an image file (supports JPG, JPEG, PNG).

3. The application will analyze the selected image and display the results with detected objects highlighted by bounding boxes.

## Contributing

Contributions are welcome! Fork the repository and submit a pull request with your enhancements.

## License

This project is licensed under the [MIT License](LICENSE).
