# The Sign Language Translator

AI-powered American Sign Language (ASL) to text translation system that helps bridge communication gaps between the deaf/hard-of-hearing community and others.

## Features

- Real-time ASL hand sign recognition
- Support for 24 ASL letters (A-Y, excluding J and Z)
- High accuracy (99.80% on test set)
- User-friendly web interface
- Confidence score display for predictions

## Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python, TensorFlow/Keras
- **Data Processing:** OpenCV, NumPy, Pandas
- **Model:** Convolutional Neural Network (CNN)

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd signvision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
streamlit run app.py
```

The application will open in your default web browser. Upload an image of an ASL hand sign, and the model will predict the corresponding letter.

## Model Architecture

- Input Layer: 28x28x1 (grayscale images)
- Multiple Convolutional Layers with BatchNormalization and MaxPooling
- Dense Layers with Dropout for regularization
- Output Layer: 24 classes (ASL letters A-Y, excluding J and Z)

## Dataset

The model is trained on the Sign Language MNIST dataset from Kaggle, which contains 27,455 training and 7,172 test images.

## Performance

- Test Accuracy: 99.80%
- Support for real-time processing
- Robust to various hand sizes and positions

## Future Enhancements

- Support for complete ASL alphabet including J and Z
- Gesture recognition capabilities
- Sentence-level translation
- Mobile application development
- Enhanced UI/UX features

## Team
- Sakshi Goswami-E23CSEU1805
- Nishtha Sikri-E23CSEU1809

## License

[Choose an appropriate license]

## Acknowledgments

- Sign Language MNIST dataset from Kaggle
- [Add any other acknowledgments] 
