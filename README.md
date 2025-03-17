# VISI0NARIES
Team V!SI0NARIES - Problem Statement 2 - Caterpillar Tech Challenge 2025

## Team Information
- **Team Name:** V!SI0NARIES
- **Project Title:** Create Vision system to identify and display defects on components

## Problem Statement
Visual inspection on Paint aesthetics, Weld aesthetics, gap measurements are currently manual posing issues on repeatability and dependence on manual processes. Technology solution needed to resolve the industry pain.

## Solution Overview
Create an AI-based Vision system with a built-in Deep Learning model to detect, identify, and display defects on components which can qualify Paint appearance, Weld appearance, assembly quality, and machining quality.

## Dataset Information
We used the [Weld Defect Detection Dataset](https://www.kaggle.com/datasets/sukmaadhiwijaya/weld-defect-detection-dataset) from Kaggle for training our model.

## Model Details
We have developed a machine learning model to detect weld defects. The model was trained using an RTX 3050 (6GB) laptop GPU and an i5-12450H CPU.

## Deployment on Raspberry Pi
We are importing the trained deep learning model on a Raspberry Pi 5 and using a Pi Camera module to create a real-time machine vision system.

### Hardware Requirements
- Raspberry Pi 5
- Pi Camera Module
- Power supply for Raspberry Pi
- SD card with Raspberry Pi OS installed

### Installation Instructions
1. Clone the repository
    ```bash
    git clone https://github.com/Kamalesh-Git/VISI0NARIES.git
    cd VISI0NARIES
    ```
2. Install necessary dependencies
    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install python3-pip
    pip3 install -r requirements.txt
    ```

### Usage Instructions
1. Connect the Pi Camera module to the Raspberry Pi.
2. Ensure the camera is enabled in the Raspberry Pi configuration.
3. Prepare the dataset (if needed).
4. Run the inference script to start the real-time defect detection.
    ```bash
    python3 infer_rpi.py
    ```

## Results
Include here the results of your model, such as accuracy, precision, recall, and any relevant visualizations.

## Future Work
- Improve the dataset with more diverse samples.
- Enhance the model to detect other types of defects.
- Integrate the model into a more robust and scalable real-time vision system.

## Contributing
Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to Kaggle for providing the dataset.
- Special thanks to the Caterpillar Tech Challenge 2025 organizers.
