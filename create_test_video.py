import cv2
import numpy as np
import urllib.request
import os

def create_synthetic_zigzag_video(output_path="evidence/synthetic_zigzag.mp4", duration_sec=10, fps=25):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Download a top-down car image to use
    car_img_path = "car_top_down.png"
    if not os.path.exists(car_img_path):
        print("Downloading car sprite...")
        url = "https://raw.githubusercontent.com/jungeuler/Autonomous-Car/master/car.png"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(car_img_path, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Failed to download car image: {e}")
            # Create a fallback colored rectangle that looks roughly like a car
            car_img = np.zeros((100, 50, 3), dtype=np.uint8)
            car_img[:] = (0, 0, 255) # Red car
            cv2.imwrite(car_img_path, car_img)

    car_sprite = cv2.imread(car_img_path)
    if car_sprite is None:
        car_sprite = np.zeros((100, 50, 3), dtype=np.uint8)
        car_sprite[:] = (0, 0, 255)
    
    # Resize car to reasonable CCTV size
    car_sprite = cv2.resize(car_sprite, (80, 160))
    car_h, car_w = car_sprite.shape[:2]

    # Create background (dark gray road)
    bg = np.ones((height, width, 3), dtype=np.uint8) * 60
    # Draw lane lines
    for x in [320, 640, 960]:
        for y in range(0, height, 100):
            cv2.rectangle(bg, (x-5, y), (x+5, y+50), (255, 255, 255), -1)

    total_frames = duration_sec * fps
    
    # Base speed moving forward (upwards in frame)
    speed_y = 5.0 
    
    for i in range(total_frames):
        frame = bg.copy()
        
        # Calculate car position
        # Start at bottom center
        y = int(height - (i * speed_y))
        
        # Sine wave for zigzag (weaving heavily across lanes)
        # 1.5 full weaves over the 10 seconds
        amplitude = 300 # pixels left/right
        frequency = 2 * np.pi * 1.5 / total_frames
        
        x = int(width / 2 + amplitude * np.sin(frequency * i))
        
        # Draw car on frame
        top_y = max(0, y - car_h//2)
        bottom_y = min(height, y + car_h//2)
        left_x = max(0, x - car_w//2)
        right_x = min(width, x + car_w//2)
        
        # Handle boundaries
        if top_y < bottom_y and left_x < right_x:
            sprite_roi = car_sprite[
                (car_h//2 - (y - top_y)):(car_h//2 + (bottom_y - y)),
                (car_w//2 - (x - left_x)):(car_w//2 + (right_x - x))
            ]
            frame[top_y:bottom_y, left_x:right_x] = sprite_roi

        out.write(frame)
        
        if y < -car_h:
            break

    out.release()
    print(f"Created synthetic test video: {output_path}")

if __name__ == "__main__":
    create_synthetic_zigzag_video()
