
# Self-Driving Car System: Lane Detection + Vehicle Detection
# Combines OpenCV lane detection with YOLOv8 vehicle detection

import cv2
import numpy as np
from ultralytics import YOLO
import math

class SelfDrivingCarSystem:
    def __init__(self):
        # Load YOLO model
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Vehicle classes we care about (from COCO dataset)
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Lane detection parameters
        self.lane_params = {
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 50,
            'hough_min_line_length': 100,
            'hough_max_line_gap': 5
        }
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for lane detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blur, self.lane_params['canny_low'], self.lane_params['canny_high'])
        
        return edges
    
    def region_of_interest(self, edges, vertices):
        """
        Apply region of interest mask for lane detection
        """
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        return masked_edges
    
    def detect_lanes(self, frame):
        """
        Detect lane lines using Hough Transform
        """
        height, width = frame.shape[:2]
        
        # Preprocess
        edges = self.preprocess_frame(frame)
        
        # Define region of interest (trapezoid focusing on road)
        vertices = np.array([[
            (0, height),
            (width // 2 - 50, height // 2 + 50),
            (width // 2 + 50, height // 2 + 50),
            (width, height)
        ]], dtype=np.int32)
        
        # Apply region of interest
        masked_edges = self.region_of_interest(edges, vertices)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            1,  # rho
            np.pi/180,  # theta
            threshold=self.lane_params['hough_threshold'],
            minLineLength=self.lane_params['hough_min_line_length'],
            maxLineGap=self.lane_params['hough_max_line_gap']
        )
        
        return lines, masked_edges
    
    def separate_lanes(self, lines, frame_width):
        """
        Separate detected lines into left and right lanes
        """
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate slope
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Left lane (negative slope)
                    if slope < -0.5 and x1 < frame_width // 2:
                        left_lines.append(line[0])
                    
                    # Right lane (positive slope)
                    elif slope > 0.5 and x1 > frame_width // 2:
                        right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def get_lane_line(self, lines, frame_height):
        """
        Get average lane line from multiple detected lines
        """
        if len(lines) == 0:
            return None
        
        # Convert to numpy array for easier computation
        lines_array = np.array(lines)
        
        # Calculate slopes and intercepts
        slopes = []
        intercepts = []
        
        for x1, y1, x2, y2 in lines_array:
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                slopes.append(slope)
                intercepts.append(intercept)
        
        if len(slopes) == 0:
            return None
        
        # Average slope and intercept
        avg_slope = np.mean(slopes)
        avg_intercept = np.mean(intercepts)
        
        # Calculate line endpoints
        y1 = frame_height
        y2 = frame_height // 2
        
        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)
        
        return [[x1, y1, x2, y2]]
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles using YOLO
        """
        results = self.yolo_model(frame, verbose=False)
        
        vehicles = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    
                    # Only process vehicle classes
                    if class_id in self.vehicle_classes:
                        confidence = float(box.conf[0])
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        vehicles.append({
                            'class': self.vehicle_classes[class_id],
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
        
        return vehicles
    
    def calculate_distance_to_vehicle(self, bbox, frame_height):
        """
        Estimate distance to vehicle based on bounding box size
        Simple approximation - larger boxes = closer vehicles
        """
        x1, y1, x2, y2 = bbox
        box_height = y2 - y1
        box_bottom = y2
        
        # Simple distance estimation
        # Vehicles closer to bottom of frame and larger are closer
        distance_factor = (frame_height - box_bottom) + (200 - box_height)
        
        if distance_factor <= 50:
            return "VERY CLOSE"
        elif distance_factor <= 100:
            return "CLOSE" 
        elif distance_factor <= 200:
            return "MEDIUM"
        else:
            return "FAR"
    
    def draw_lanes(self, frame, left_lane, right_lane):
        """
        Draw detected lanes on frame
        """
        lane_frame = np.zeros_like(frame)
        
        # Draw left lane
        if left_lane is not None:
            cv2.line(lane_frame, 
                    (left_lane[0][0], left_lane[0][1]), 
                    (left_lane[0][2], left_lane[0][3]), 
                    (0, 255, 0), 8)
        
        # Draw right lane
        if right_lane is not None:
            cv2.line(lane_frame, 
                    (right_lane[0][0], right_lane[0][1]), 
                    (right_lane[0][2], right_lane[0][3]), 
                    (0, 255, 0), 8)
        
        # Fill the lane area
        if left_lane is not None and right_lane is not None:
            pts = np.array([
                [left_lane[0][0], left_lane[0][1]],
                [left_lane[0][2], left_lane[0][3]], 
                [right_lane[0][2], right_lane[0][3]],
                [right_lane[0][0], right_lane[0][1]]
            ], np.int32)
            
            cv2.fillPoly(lane_frame, [pts], (0, 255, 0))
        
        # Combine with original frame
        result = cv2.addWeighted(frame, 0.8, lane_frame, 0.3, 0)
        return result
    
    def draw_vehicles(self, frame, vehicles):
        """
        Draw detected vehicles with distance information
        """
        for vehicle in vehicles:
            bbox = vehicle['bbox']
            x1, y1, x2, y2 = bbox
            
            # Determine box color based on distance
            distance = self.calculate_distance_to_vehicle(bbox, frame.shape[0])
            
            if distance == "VERY CLOSE":
                color = (0, 0, 255)  # Red
            elif distance == "CLOSE":
                color = (0, 165, 255)  # Orange
            elif distance == "MEDIUM":
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{vehicle['class']} ({vehicle['confidence']:.2f}) - {distance}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def add_dashboard(self, frame, vehicles, left_lane, right_lane):
        """
        Add driving dashboard with information
        """
        height, width = frame.shape[:2]
        
        # Create dashboard area
        dashboard = np.zeros((150, width, 3), dtype=np.uint8)
        
        # Lane status
        left_status = "✓" if left_lane is not None else "✗"
        right_status = "✓" if right_lane is not None else "✗"
        
        cv2.putText(dashboard, f"Left Lane: {left_status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if left_lane else (0, 0, 255), 2)
        
        cv2.putText(dashboard, f"Right Lane: {right_status}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if right_lane else (0, 0, 255), 2)
        
        # Vehicle count by distance
        close_vehicles = sum(1 for v in vehicles if self.calculate_distance_to_vehicle(v['bbox'], height) in ["VERY CLOSE", "CLOSE"])
        total_vehicles = len(vehicles)
        
        cv2.putText(dashboard, f"Total Vehicles: {total_vehicles}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(dashboard, f"Close Vehicles: {close_vehicles}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255) if close_vehicles > 0 else (0, 255, 0), 2)
        
        # Driving recommendation
        if close_vehicles > 0:
            recommendation = "SLOW DOWN - VEHICLE TOO CLOSE"
            color = (0, 0, 255)
        elif left_lane is None or right_lane is None:
            recommendation = "CAUTION - LANE NOT DETECTED"
            color = (0, 255, 255)
        else:
            recommendation = "SAFE TO DRIVE"
            color = (0, 255, 0)
        
        cv2.putText(dashboard, recommendation, (width//2 - 100, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Combine dashboard with frame
        combined = np.vstack((frame, dashboard))
        return combined
    
    def process_frame(self, frame):
        """
        Main function to process each frame
        """
        # Make a copy to work with
        result_frame = frame.copy()
        
        # 1. Detect lanes
        lines, masked_edges = self.detect_lanes(frame)
        left_lines, right_lines = self.separate_lanes(lines, frame.shape[1])
        
        # Get lane lines
        left_lane = self.get_lane_line(left_lines, frame.shape[0])
        right_lane = self.get_lane_line(right_lines, frame.shape[0])
        
        # 2. Detect vehicles
        vehicles = self.detect_vehicles(frame)
        
        # 3. Draw lanes
        result_frame = self.draw_lanes(result_frame, left_lane, right_lane)
        
        # 4. Draw vehicles
        result_frame = self.draw_vehicles(result_frame, vehicles)
        
        # 5. Add dashboard
        result_frame = self.add_dashboard(result_frame, vehicles, left_lane, right_lane)
        
        return result_frame, masked_edges

# ========================================
# USAGE FUNCTIONS
# ========================================

def process_video(video_path, output_path=None):
    """
    Process a video file with self-driving car system
    """
    system = SelfDrivingCarSystem()
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 150))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, edges = system.process_frame(frame)
        
        # Write to output video
        if output_path:
            out.write(processed_frame)
        
        # Display progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
        
        # Optional: display frame (comment out for faster processing)
        # cv2.imshow('Self-Driving Car System', processed_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        frame_count += 1
    
    cap.release()
    if output_path:
        out.release()
        print(f"Output saved to: {output_path}")
    
    cv2.destroyAllWindows()

def process_webcam():
    """
    Real-time processing using webcam
    """
    system = SelfDrivingCarSystem()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    print("Real-time self-driving car system started.")
    print("Press 'q' to quit, 's' to save current frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, edges = system.process_frame(frame)
        
        # Display
        cv2.imshow('Self-Driving Car System', processed_frame)
        cv2.imshow('Lane Detection (Edges)', edges)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('self_driving_snapshot.jpg', processed_frame)
            print("Frame saved as self_driving_snapshot.jpg")
    
    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    """
    Process a single image
    """
    system = SelfDrivingCarSystem()
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Process
    result, edges = system.process_frame(frame)
    
    # Save results
    cv2.imwrite('self_driving_result.jpg', result)
    cv2.imwrite('lane_edges.jpg', edges)
    
    print("Results saved:")
    print("- self_driving_result.jpg (full system output)")
    print("- lane_edges.jpg (lane detection edges)")

# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("Self-Driving Car System")
    print("="*50)
    print("Choose an option:")
    print("1. Process webcam (real-time)")
    print("2. Process image file") 
    print("3. Process video file")
    
    choice = input("Enter choice (1/2/3): ")
    
    if choice == "1":
        process_webcam()
    
    elif choice == "2":
        image_path = input("Enter image path: ")
        process_image(image_path)
    
    elif choice == "3":
        video_path = input("Enter video path: ")
        output_path = input("Enter output path (optional, press Enter to skip): ")
        if output_path == "":
            output_path = None
        process_video(video_path, output_path)
    
    else:
        print("Invalid choice")