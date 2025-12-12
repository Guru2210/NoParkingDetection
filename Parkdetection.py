import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from scipy.spatial import distance as dist
from collections import OrderedDict
from centroid_tracker import CentroidTracker

# Load YOLO models for vehicle and license plate detection
license_plate_model = YOLO("NumberPlateDetection.pt")  # License plate detection model

model = YOLO("yolov8n.pt")
tracker = CentroidTracker(maxDisappeared=5)
vehicle_positions = {}
position_threshold = 20  # pixels
time_threshold = 10   # seconds
vehicle_saved = set()  # To track which vehicles have been saved

def detect_license_plate(frame):
    """Detect license plates in the given frame."""
    results = license_plate_model(frame)
    plates = []

    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = result
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        plates.append((bbox, conf))

    return plates

def detect_vehicles(frame):
    results = model(frame)
    vehicle_classes = [2, 3, 5, 7]
    rects = []

    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = result
        if int(class_id) in vehicle_classes:
            rects.append((int(x1), int(y1), int(x2), int(y2)))

    objects = tracker.update(rects)
    current_time = time.time()

    for objectID in objects.keys():
        centroid = objects[objectID]
        if objectID not in vehicle_positions:
            vehicle_positions[objectID] = {'centroid': centroid, 'start_time': current_time}
        else:
            prev_centroid = vehicle_positions[objectID]['centroid']
            if dist.euclidean(centroid, prev_centroid) < position_threshold:
                elapsed_time = current_time - vehicle_positions[objectID]['start_time']
                if elapsed_time > time_threshold and objectID not in vehicle_saved:
                    vehicle_saved.add(objectID)  # Mark this vehicle as saved
                    cv2.putText(frame, f"Unauthorized Parking!", (centroid[0] - 20, centroid[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    x1, y1, x2, y2 = rects[objectID % len(rects)]
                    cropped_vehicle = frame[y1:y2, x1:x2]
                    save_vehicle_and_license_plate(cropped_vehicle, objectID)
            else:
                vehicle_positions[objectID]['centroid'] = centroid
                vehicle_positions[objectID]['start_time'] = current_time

        x1, y1, x2, y2 = rects[objectID % len(rects)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def save_vehicle_and_license_plate(cropped_vehicle, track_id):
    """Save the cropped image of the vehicle and detected license plate."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Detect license plate in the vehicle region
    plate_detections = detect_license_plate(cropped_vehicle)

    for plate_bbox, conf in plate_detections:
        px1, py1, px2, py2 = plate_bbox
        cv2.rectangle(cropped_vehicle, (px1, py1), (px2, py2), (0, 0, 255), 2)
        
    # Save the image with both vehicle and license plate details
    filename = f"unauthorized_vehicle_{track_id}_{timestamp}.jpg"
    cv2.imwrite(filename, cropped_vehicle)
    print(f"Unauthorized parking detected. Image saved: {filename}")

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get the frame width, height, and FPS for the video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create a VideoWriter object to save the output video in MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detect_vehicles(frame)

        # Write the frame with annotations to the output video
        out.write(frame)
        
        # Show the frame
        cv2.imshow("Parking and License Plate Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "parking_lot_video3.mp4"  # Replace with your video path
    output_path = "parkingDetectionWithNumberPlate_CentroidTracker.mp4"  # Output video path
    process_video(video_path, output_path)
