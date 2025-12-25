Face Mask Detection System (End-to-End Pipeline)

An end-to-end Face Mask Detection System built using Deep Learning (CNN) that detects whether a person is wearing a mask, not wearing a mask, or wearing it incorrectly. The system supports image upload, webcam-based detection, and edge deployment using TensorFlow Lite, along with a CI pipeline using GitHub Actions.

Features
- CNN-based face mask classification
- MobileNetV2 as pre-trained feature extractor
- Classes: with_mask, without_mask, mask_weared_incorrect
- Face detection using OpenCV Haar Cascade
- Image upload and webcam support
- Deployment using OpenCV, Streamlit, and TFLite
- CI pipeline using GitHub Actions

Model Architecture
- Backbone: MobileNetV2 (ImageNet)
- Heads: Classification + Bounding Box Regression
- Loss: Categorical Cross-Entropy + MSE

End-to-End Pipeline
Input Image/Webcam → Preprocessing → Face Detection → CNN Feature Extraction → Classification → Bounding Box Visualization → Deployment

Deployment
- OpenCV webcam-based detection (local)
- Streamlit web application
- TensorFlow Lite for edge devices

CI/CD GitHub Actions pipeline validates model files, runs inference tests, and checks Streamlit app syntax.

License
Educational and academic use only.