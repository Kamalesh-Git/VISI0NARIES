# Step 1: Import necessary modules
from ultralytics import YOLO
import cv2
import torch
import os

# =====================================================
# Step 2: Configure dataset paths (MODIFY THESE VARIABLES)
# =====================================================
BASE_DIR = r"D:\VISI0NARIES\data\dataset"  # <--- UPDATE THIS TO YOUR DATASET PATH
TRAIN_IMAGES = os.path.join(BASE_DIR, "train/images")
TRAIN_LABELS = os.path.join(BASE_DIR, "train/labels")
VALID_IMAGES = os.path.join(BASE_DIR, "valid/images")
VALID_LABELS = os.path.join(BASE_DIR, "valid/labels")
TEST_IMAGES = os.path.join(BASE_DIR, "test/images")
TEST_LABELS = os.path.join(BASE_DIR, "test/labels")

# Create data.yaml configuration file programmatically
def create_data_yaml():
    data_config = f"""
    train: {TRAIN_IMAGES}
    val: {VALID_IMAGES}
    test: {TEST_IMAGES}

    nc: 6
    names: ['Bad Welding', 'Crack', 'Excess Reinforcement', 'Good Welding', 'Porosity', 'Spatters']
    """
    with open("custom_data.yaml", "w") as f:
        f.write(data_config.strip())
    print(f"Created custom_data.yaml with:\n{data_config}")

# Initialize configuration
create_data_yaml()

# =====================================================
# Step 3: Optimized Training Configuration
# =====================================================
def train_model():
    # Ensure GPU is used
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    model = YOLO('yolov8n.pt')  # Start with pretrained weights
    
    # Optimize for RTX 3050 (6GB VRAM)
    results = model.train(
        data="custom_data.yaml",  # Uses our generated config
        epochs=150,               # Increase epochs for better convergence
        imgsz=640,                # Balance between resolution and memory usage
        batch=8,                  # Batch size optimized for 6GB VRAM
        device=device,            # Explicitly set device (0 for GPU)
        workers=4,                # Number of CPU threads for data loading
        project='welding_defects',
        name='gpu_training',
        val=True,
        augment=True,             # Enable data augmentation
        cache=True,               # Cache images in memory for faster training
        amp=True,                 # Mixed precision training to reduce memory usage
        lr0=0.01,                 # Initial learning rate
        optimizer='AdamW',        # AdamW optimizer for better convergence
        close_mosaic=10,          # Disable mosaic augmentation in the last 10 epochs
        class_weights=[1.0, 2.0, 1.5, 0.5, 1.2, 1.0]  # Handle class imbalance
    )
    return model

# =====================================================
# Step 4: Evaluation and Inference
# =====================================================
def evaluate_model(model):
    metrics = model.val()
    print(f"\nValidation Results:")
    print(f"mAP50-95: {metrics.box.map:.2f}")
    print(f"mAP50: {metrics.box.map50:.2f}")
    print(f"Precision: {metrics.box.mp:.2f}")
    print(f"Recall: {metrics.box.mr:.2f}")

def detect_and_classify(model, img_path, conf=0.5):
    # Ensure GPU is used
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    results = model(img_path, conf=conf, device=device)
    
    # Class indices for defects (0-5, excluding 3: Good Welding)
    defect_classes = {0, 1, 2, 4, 5}
    detected_classes = set()
    
    for box in results[0].boxes:
        detected_classes.add(int(box.cls.item()))
    
    has_defect = not detected_classes.isdisjoint(defect_classes)
    
    # Show results
    results[0].show()
    print("\nDetection Results:")
    print(f"Detected classes: {[model.names[c] for c in detected_classes]}")
    print("Defect Detected" if has_defect else "No Defect Found")

# =====================================================
# Step 5: Real-time Detection
# =====================================================
def real_time_detection(model, conf=0.5):
    # Ensure GPU is used
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    cap = cv2.VideoCapture(0)
    defect_classes = {0, 1, 2, 4, 5}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=conf, device=device)
        annotated_frame = results[0].plot()
        
        # Check for defects
        detected_classes = set()
        for box in results[0].boxes:
            detected_classes.add(int(box.cls.item()))
        
        has_defect = not detected_classes.isdisjoint(defect_classes)
        
        # Add status text
        status_text = "Defect Detected" if has_defect else "No Defect"
        color = (0, 0, 255) if has_defect else (0, 255, 0)
        cv2.putText(annotated_frame, status_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Show FPS
        cv2.putText(annotated_frame, f"Conf: {conf:.2f}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow('Welding Defect Detection', annotated_frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('+'):
            conf = min(1.0, conf + 0.05)
        elif key == ord('-'):
            conf = max(0.05, conf - 0.05)
    
    cap.release()
    cv2.destroyAllWindows()

# =====================================================
# Main Execution
# =====================================================
if __name__ == "__main__":
    # Verify paths
    assert os.path.exists(TRAIN_IMAGES), f"Training images path not found: {TRAIN_IMAGES}"
    assert os.path.exists(VALID_IMAGES), f"Validation images path not found: {VALID_IMAGES}"
    
    # Train model (uncomment to train)
    model = train_model()  # <--- UNCOMMENT THIS LINE TO START TRAINING
    
    # Load trained model
    # model = YOLO('welding_defects/gpu_training/weights/best.pt')  # <--- UPDATE IF NEEDED
    
    # Evaluate
    evaluate_model(model)
    
    # Test detection
    test_image = os.path.join(TEST_IMAGES, "sample.jpg")  # Update with your test image
    detect_and_classify(model, test_image, conf=0.6)
    
    # Real-time detection (uncomment to use)
    # real_time_detection(model, conf=0.6)