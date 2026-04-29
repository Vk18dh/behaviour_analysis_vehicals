import cv2
import numpy as np
from ultralytics import YOLO

def generate_synthetic_weaving_video():
    # Load YOLO to find a real car crop
    model = YOLO("yolov8n.pt")
    
    cap = cv2.VideoCapture("evidence/test_cctv_traffic.mp4")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read CCTV video")
        return

    # Detect cars in the first frame
    results = model(frame, verbose=False)
    car_crop = None
    
    for box in results[0].boxes:
        # Class 2 is car in COCO
        if int(box.cls[0]) == 2:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Ensure crop is decent size
            if (x2 - x1) > 40 and (y2 - y1) > 40:
                car_crop = frame[y1:y2, x1:x2]
                break
                
    if car_crop is None:
        # Fallback to drawing a simple blocky car if none found
        car_crop = np.zeros((80, 40, 3), dtype=np.uint8)
        car_crop[:] = (50, 50, 200)

    # Now create the synthetic video
    width, height = 1280, 720
    fps = 25
    duration = 10
    total_frames = duration * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("evidence/extreme_zigzag.mp4", fourcc, fps, (width, height))
    
    # Static grey background
    bg = np.ones((height, width, 3), dtype=np.uint8) * 100
    
    # Draw simple white dashed lane lines
    for x in [400, 640, 880]:
        for y in range(0, height, 80):
            cv2.rectangle(bg, (x-5, y), (x+5, y+40), (255, 255, 255), -1)

    car_h, car_w = car_crop.shape[:2]
    
    for i in range(total_frames):
        frame_canvas = bg.copy()
        
        # Move from top to bottom
        progress = i / total_frames
        y_center = int(progress * (height + car_h*2) - car_h)
        
        # Violent zigzag (sine wave)
        # 3 full weaves across a 600 pixel amplitude
        x_center = int(width/2 + 300 * np.sin(progress * 3 * 2 * np.pi))
        
        y1 = max(0, y_center - car_h//2)
        y2 = min(height, y_center + car_h//2)
        x1 = max(0, x_center - car_w//2)
        x2 = min(width, x_center + car_w//2)
        
        # Handle edges
        if y1 < y2 and x1 < x2:
            crop_y1 = car_h//2 - (y_center - y1)
            crop_y2 = car_h//2 + (y2 - y_center)
            crop_x1 = car_w//2 - (x_center - x1)
            crop_x2 = car_w//2 + (x2 - x_center)
            
            frame_canvas[y1:y2, x1:x2] = car_crop[crop_y1:crop_y2, crop_x1:crop_x2]
            
        out.write(frame_canvas)
        
    out.release()
    print("Successfully generated evidence/extreme_zigzag.mp4")

if __name__ == "__main__":
    generate_synthetic_weaving_video()
