import cv2
import numpy as np
from ultralytics import YOLO
import time
import requests
from datetime import datetime
from enum import Enum
import threading
import queue
import os

# ===== CONFIGURATION =====
class Config:
    # Camera settings
    CAMERA_WIDTH = 320
    CAMERA_HEIGHT = 240
    TARGET_FPS = 5
    DISPLAY_SCALE = 3.0
    
    # Timing settings
    SWITCH_DELAY = 1.0      # Wait before switching cameras
    DETECTION_DELAY = 2.0   # Wait after switch before detecting
    
    # NEW: Confirmation settings
    CONFIRMATION_FRAMES = 2      # Must detect for 3 consecutive frames
    CONFIRMATION_MIN_TIME = 0.1  # Must detect for at least 1 second
    
    # Detection settings
    ROI_LINE_POSITION = 0.5  # Middle of frame
    ROI_THRESHOLD = 15       # Pixels around ROI line
    
    # Class IDs for 6-class model
    SIDE_DENT = 0
    SIDE_HOLE = 1
    SIDE_PRODUCT = 2
    TOP_DENT = 3
    TOP_HOLE = 4
    TOP_PRODUCT = 5

class CameraState(Enum):
    CAMERA_0_ACTIVE = 0  # Top camera
    CAMERA_1_ACTIVE = 1  # Side camera

# ===== NEW: CONFIRMATION TRACKER =====
class ConfirmationTracker:
    def __init__(self, required_frames=3, min_time=1.0):
        self.required_frames = required_frames
        self.min_time = min_time
        self.reset()
    
    def reset(self):
        self.consecutive_frames = 0
        self.first_detection_time = None
        self.confirmed = False
    
    def update(self, detected):
        current_time = time.time()
        
        if detected:
            if self.first_detection_time is None:
                self.first_detection_time = current_time
                self.consecutive_frames = 1
            else:
                self.consecutive_frames += 1
            
            # Check if confirmation criteria met
            time_elapsed = current_time - self.first_detection_time
            if (self.consecutive_frames >= self.required_frames and 
                time_elapsed >= self.min_time):
                self.confirmed = True
        else:
            # Reset if detection lost
            self.reset()
        
        return self.confirmed
    
    def get_progress(self):
        if self.first_detection_time is None:
            return {"frames": 0, "time": 0.0, "frames_needed": self.required_frames, "time_needed": self.min_time}
        
        current_time = time.time()
        time_elapsed = current_time - self.first_detection_time
        
        return {
            "frames": self.consecutive_frames,
            "time": time_elapsed,
            "frames_needed": self.required_frames,
            "time_needed": self.min_time,
            "frames_ok": self.consecutive_frames >= self.required_frames,
            "time_ok": time_elapsed >= self.min_time
        }
    
    def is_confirming(self):
        return self.first_detection_time is not None and not self.confirmed

# ===== NON-BLOCKING BACKEND SENDER =====
class BackendSender:
    def __init__(self, url="https://your-backend-url.vercel.app"):
        self.url = url.rstrip('/')
        self.connected = self._test_connection()
        
        # Threading for non-blocking requests
        self.send_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        # Track send status
        self.last_send_status = "Ready"
        self.sends_today = 0
    
    def _test_connection(self):
        try:
            response = requests.get(self.url, timeout=1)
            if response.status_code == 200:
                print("Backend: Connected")
                return True
        except:
            pass
        print("Backend: Offline")
        return False
    
    def _worker(self):
        """Background thread that handles all HTTP requests"""
        while True:
            try:
                data = self.send_queue.get(timeout=1)
                if data is None:  # Shutdown signal
                    break
                
                self.last_send_status = "Sending..."
                start_time = time.time()
                
                try:
                    response = requests.post(f"{self.url}/api/data", json=data, timeout=5)
                    send_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        self.last_send_status = f"Sent OK ({send_time:.1f}s)"
                        self.sends_today += 1
                        print(f"? Sent: Product #{data['payload']['product_id']} - {data['payload']['status']}")
                    else:
                        self.last_send_status = f"Failed: {response.status_code}"
                        print(f"? Failed: Product #{data['payload']['product_id']} - Status {response.status_code}")
                
                except requests.exceptions.Timeout:
                    self.last_send_status = "Failed: Timeout"
                    print(f"? Timeout: Product #{data['payload']['product_id']}")
                except Exception as e:
                    self.last_send_status = f"Failed: {str(e)[:20]}"
                    print(f"? Error: Product #{data['payload']['product_id']} - {e}")
                
                self.send_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker thread error: {e}")
    
    def send_product(self, product_id, status, holes_count, processing_time):
        """Non-blocking send - adds to queue and returns immediately"""
        if not self.connected:
            self.last_send_status = "Backend Offline"
            return
        
        defects = ['hole'] * holes_count if holes_count > 0 else []
        data = {
            "type": "product",
            "payload": {
                "product_id": product_id,
                "status": status,
                "processing_time": round(processing_time, 2),
                "defects": defects,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add to queue - this is instant and non-blocking
        self.send_queue.put(data)
        self.last_send_status = "Queued"
    
    def get_status(self):
        queue_size = self.send_queue.qsize()
        return {
            'status': self.last_send_status,
            'queue_size': queue_size,
            'sends_today': self.sends_today,
            'connected': self.connected
        }
    
    def shutdown(self):
        """Clean shutdown of worker thread"""
        self.send_queue.put(None)
        self.worker_thread.join(timeout=2)

# ===== ENHANCED PRODUCT TRACKER WITH SIDE HOLE DETECTION =====
class Product:
    def __init__(self, product_id, start_time):
        self.id = product_id
        self.start_time = start_time
        self.holes_count = 0  # Only count holes from top camera
        self.has_dents = False
        self.has_side_holes = False  # NEW: Track side hole detection (but don't count)
        self.is_complete = False
    
    def add_holes(self, count):
        """Add holes from top camera (counted)"""
        self.holes_count = max(self.holes_count, count)
    
    def add_side_holes(self):
        """Mark that side holes were detected (not counted)"""
        self.has_side_holes = True
    
    def add_dents(self):
        """Mark that dents were detected"""
        self.has_dents = True
    
    def get_status(self):
        """Generate status string including side holes"""
        defects = []
        
        # Add hole count from top camera
        if self.holes_count > 0:
            defects.append(f"{self.holes_count} hole{'s' if self.holes_count > 1 else ''}")
        
        # Add side holes detection (without count)
        elif self.has_side_holes and self.holes_count == 0:
            defects.append("holes (side)")
        
        # Add dents
        if self.has_dents:
            defects.append("dents")
        
        if defects:
            return f"DEFECT ({', '.join(defects)})"
        else:
            return "OK"
    
    def is_defective(self):
        """Check if product is defective (includes side holes)"""
        return self.holes_count > 0 or self.has_dents or self.has_side_holes

class ProductTracker:
    def __init__(self, backend_sender=None):
        self.backend = backend_sender
        self.product_counter = 0
        self.current_product = None
        self.completed_products = []
    
    def start_new_product(self):
        self.product_counter += 1
        self.current_product = Product(self.product_counter, time.time())
        print(f"?? CONFIRMED: Started Product #{self.current_product.id}")
        return self.current_product
    
    def update_from_top_camera(self, holes, dents):
        """Update from top camera - count holes and detect dents"""
        if self.current_product:
            if holes > 0:
                self.current_product.add_holes(holes)
                print(f"?? Top Camera: Product #{self.current_product.id} - {holes} holes COUNTED")
            if dents > 0:
                self.current_product.add_dents()
                print(f"?? Top Camera: Product #{self.current_product.id} - dents detected")
    
    def update_from_side_camera(self, holes, dents):
        """Update from side camera - detect holes (don't count) and dents"""
        if self.current_product:
            # Side camera detects holes but doesn't count them
            if holes > 0:
                self.current_product.add_side_holes()
                print(f"?? Side Camera: Product #{self.current_product.id} - {holes} holes DETECTED (not counted)")
            if dents > 0:
                self.current_product.add_dents()
                print(f"?? Side Camera: Product #{self.current_product.id} - dents detected")
    
    def finish_current_product(self):
        if not self.current_product:
            return None
        
        processing_time = time.time() - self.current_product.start_time
        status = self.current_product.get_status()
        
        # Send to backend (now non-blocking!)
        if self.backend:
            backend_status = "DEFECTIVE" if self.current_product.is_defective() else "OK"
            # Only send counted holes (from top camera) to backend
            self.backend.send_product(
                self.current_product.id,
                backend_status,
                self.current_product.holes_count,  # Only top camera holes
                processing_time
            )
        
        self.completed_products.append(self.current_product)
        print(f"? CONFIRMED: Completed Product #{self.current_product.id} - {status}")
        
        finished_product = self.current_product
        self.current_product = None
        return finished_product
    
    def get_stats(self):
        total = len(self.completed_products)
        defective = sum(1 for p in self.completed_products if p.is_defective())
        total_holes = sum(p.holes_count for p in self.completed_products)  # Only counted holes
        side_hole_detections = sum(1 for p in self.completed_products if p.has_side_holes)
        
        return {
            'total': total,
            'ok': total - defective,
            'defective': defective,
            'total_holes': total_holes,  # Only from top camera
            'side_hole_detections': side_hole_detections,  # Side detections
            'current_id': self.current_product.id if self.current_product else None
        }

# ===== ENHANCED CAMERA CONTROLLER =====
class CameraController:
    def __init__(self):
        self.state = CameraState.CAMERA_0_ACTIVE
        self.switch_requested_time = None
        self.last_switch_time = 0
        
        # NEW: Confirmation trackers for each camera
        self.camera_0_confirmer = ConfirmationTracker(Config.CONFIRMATION_FRAMES, Config.CONFIRMATION_MIN_TIME)
        self.camera_1_confirmer = ConfirmationTracker(Config.CONFIRMATION_FRAMES, Config.CONFIRMATION_MIN_TIME)
        
        # Initialize both cameras
        self.camera_0 = self._init_camera(0)
        self.camera_1 = self._init_camera(1)
        
        if not self.camera_0 or not self.camera_1:
            raise Exception("Failed to initialize cameras")
    
    def _init_camera(self, camera_id):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)
            ret, _ = cap.read()
            if ret:
                print(f"Camera {camera_id}: Ready")
                return cap
        print(f"Camera {camera_id}: Failed")
        return None
    
    def read_frames(self):
        # Always read both cameras to keep buffers fresh
        ret0, frame0 = self.camera_0.read()
        ret1, frame1 = self.camera_1.read()
        
        if not ret0 or not ret1:
            return None, None, None
        
        # Return active frame and camera name
        if self.state == CameraState.CAMERA_0_ACTIVE:
            return frame0, "Camera 0 (Top)", True
        else:
            return frame1, "Camera 1 (Side)", False
    
    def update_confirmation(self, product_in_roi, is_camera_0):
        """Update confirmation tracker and return if confirmed"""
        if is_camera_0:
            return self.camera_0_confirmer.update(product_in_roi)
        else:
            return self.camera_1_confirmer.update(product_in_roi)
    
    def get_confirmation_progress(self, is_camera_0):
        """Get confirmation progress for display"""
        if is_camera_0:
            return self.camera_0_confirmer.get_progress()
        else:
            return self.camera_1_confirmer.get_progress()
    
    def is_confirming(self, is_camera_0):
        """Check if currently confirming"""
        if is_camera_0:
            return self.camera_0_confirmer.is_confirming()
        else:
            return self.camera_1_confirmer.is_confirming()
    
    def reset_confirmation(self, is_camera_0):
        """Reset confirmation tracker"""
        if is_camera_0:
            self.camera_0_confirmer.reset()
        else:
            self.camera_1_confirmer.reset()
    
    def request_switch(self):
        if not self.switch_requested_time:
            self.switch_requested_time = time.time()
            next_camera = "Camera 1" if self.state == CameraState.CAMERA_0_ACTIVE else "Camera 0"
            print(f"Switch to {next_camera} in {Config.SWITCH_DELAY}s")
            
            # Reset confirmation trackers when switching
            self.camera_0_confirmer.reset()
            self.camera_1_confirmer.reset()
    
    def update(self):
        current_time = time.time()
        
        # Check if switch delay has passed
        if self.switch_requested_time:
            if current_time - self.switch_requested_time >= Config.SWITCH_DELAY:
                self._execute_switch()
        
        # Check if detection delay has passed after switch
        return current_time - self.last_switch_time >= Config.DETECTION_DELAY
    
    def get_time_until_detection(self):
        """Returns seconds remaining until detection is ready"""
        if not self.last_switch_time:
            return 0.0
        
        current_time = time.time()
        elapsed_since_switch = current_time - self.last_switch_time
        remaining = Config.DETECTION_DELAY - elapsed_since_switch
        return max(0.0, remaining)
    
    def get_time_until_switch(self):
        """Returns seconds remaining until camera switch"""
        if not self.switch_requested_time:
            return 0.0
        
        current_time = time.time()
        elapsed_since_request = current_time - self.switch_requested_time
        remaining = Config.SWITCH_DELAY - elapsed_since_request
        return max(0.0, remaining)
    
    def _execute_switch(self):
        if self.state == CameraState.CAMERA_0_ACTIVE:
            self.state = CameraState.CAMERA_1_ACTIVE
            print("Switched to: Camera 1")
        else:
            self.state = CameraState.CAMERA_0_ACTIVE
            print("Switched to: Camera 0")
        
        self.switch_requested_time = None
        self.last_switch_time = time.time()
    
    def is_camera_0_active(self):
        return self.state == CameraState.CAMERA_0_ACTIVE
    
    def get_state_name(self):
        return self.state.name
    
    def cleanup(self):
        if self.camera_0:
            self.camera_0.release()
        if self.camera_1:
            self.camera_1.release()

# ===== DETECTION PROCESSOR =====
class DetectionProcessor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.roi_x = int(Config.CAMERA_WIDTH * Config.ROI_LINE_POSITION)
        print(f"Model loaded: {len(self.model.names)} classes")
    
    def process_frame(self, frame, is_camera_0):
        results = self.model(frame)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            center_x = (x1 + x2) // 2
            
            # Filter detections based on camera
            if is_camera_0 and cls_id in [Config.TOP_PRODUCT, Config.TOP_HOLE, Config.TOP_DENT]:
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'cls_id': cls_id,
                    'conf': conf,
                    'center_x': center_x,
                    'in_roi': abs(center_x - self.roi_x) <= Config.ROI_THRESHOLD
                })
            elif not is_camera_0 and cls_id in [Config.SIDE_PRODUCT, Config.SIDE_HOLE, Config.SIDE_DENT]:
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'cls_id': cls_id,
                    'conf': conf,
                    'center_x': center_x,
                    'in_roi': abs(center_x - self.roi_x) <= Config.ROI_THRESHOLD
                })
        
        return detections, results
    
    def count_defects(self, detections, is_camera_0):
        holes = 0
        dents = 0
        
        for det in detections:
            if is_camera_0:
                if det['cls_id'] == Config.TOP_HOLE:
                    holes += 1
                elif det['cls_id'] == Config.TOP_DENT:
                    dents += 1
            else:
                if det['cls_id'] == Config.SIDE_HOLE:
                    holes += 1
                elif det['cls_id'] == Config.SIDE_DENT:
                    dents += 1
        
        return holes, dents
    
    def has_product_in_roi(self, detections, is_camera_0):
        product_class = Config.TOP_PRODUCT if is_camera_0 else Config.SIDE_PRODUCT
        return any(det['cls_id'] == product_class and det['in_roi'] for det in detections)

# ===== ENHANCED DISPLAY MANAGER =====
class DisplayManager:
    def __init__(self):
        self.display_width = int(Config.CAMERA_WIDTH * Config.DISPLAY_SCALE)
        self.display_height = int(Config.CAMERA_HEIGHT * Config.DISPLAY_SCALE)
        self.roi_x = int(Config.CAMERA_WIDTH * Config.ROI_LINE_POSITION)
    
    def draw_detections(self, frame, detections, results):
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls_id = det['cls_id']
            conf = det['conf']
            
            # Enhanced color coding
            if cls_id in [Config.TOP_PRODUCT, Config.SIDE_PRODUCT]:
                color = (0, 255, 0)  # Green for products
            elif cls_id == Config.TOP_HOLE:
                color = (0, 0, 255)  # Red for top holes (counted)
            elif cls_id == Config.SIDE_HOLE:
                color = (255, 165, 0)  # Orange for side holes (detected only)
            else:  # Dents
                color = (0, 0, 255)  # Red for dents
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Enhanced labels
            label = f'{results.names[cls_id]} {conf:.2f}'
            if cls_id == Config.SIDE_HOLE:
                label += ' (DETECTED)'
            elif cls_id == Config.TOP_HOLE:
                label += ' (COUNTED)'
            
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def draw_roi_line(self, frame, can_detect):
        color = (255, 255, 0) if can_detect else (0, 0, 255)  # Yellow if ready, red if waiting
        cv2.line(frame, (self.roi_x, 0), (self.roi_x, Config.CAMERA_HEIGHT), color, 2)
    
    def add_camera_label_with_confirmation(self, frame, camera_name, camera_controller):
        """Camera label with confirmation status"""
        # Camera name
        cv2.putText(frame, camera_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Timer display
        switch_time = camera_controller.get_time_until_switch()
        detect_time = camera_controller.get_time_until_detection()
        is_camera_0 = camera_controller.is_camera_0_active()
        
        if switch_time > 0:
            # Switching countdown
            timer_text = f"SWITCHING IN: {switch_time:.1f}s"
            color = (0, 165, 255)  # Orange
            cv2.putText(frame, timer_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif detect_time > 0:
            # Detection countdown
            timer_text = f"DETECTION IN: {detect_time:.1f}s"
            color = (0, 0, 255)  # Red
            cv2.putText(frame, timer_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif camera_controller.is_confirming(is_camera_0):
            # Simple confirmation text
            timer_text = "CONFIRMING..."
            color = (255, 255, 0)  # Yellow
            cv2.putText(frame, timer_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Ready to detect
            timer_text = "READY TO DETECT"
            color = (0, 255, 0)  # Green
            cv2.putText(frame, timer_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def create_stats_window(self, tracker, camera_controller, backend):
        img = np.zeros((700, 400, 3), dtype=np.uint8)
        stats = tracker.get_stats()
        backend_status = backend.get_status()
        
        # Title
        cv2.putText(img, 'INSPECTION SYSTEM', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(img, (10, 40), (390, 40), (255, 255, 255), 1)
        
        y = 70
        # Camera status
        camera_color = (0, 255, 0) if camera_controller.is_camera_0_active() else (255, 165, 0)
        cv2.putText(img, f'Active: {camera_controller.get_state_name()}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, camera_color, 2)
        
        
        y += 25
        # Backend status with enhanced info
        if backend_status['connected']:
            backend_color = (0, 255, 0)
            status_text = f"Backend: ONLINE ({backend_status['sends_today']} sent)"
        else:
            backend_color = (0, 0, 255)
            status_text = "Backend: OFFLINE"
        cv2.putText(img, status_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, backend_color, 2)
        
        y += 20
        # Backend send status
        send_status = backend_status['status']
        queue_size = backend_status['queue_size']
        if queue_size > 0:
            send_text = f"Queue: {queue_size} | {send_status}"
        else:
            send_text = f"Send: {send_status}"
        
        status_color = (255, 255, 0) if "Sending" in send_status else (0, 255, 0)
        if "Failed" in send_status or "Timeout" in send_status:
            status_color = (0, 0, 255)
        
        cv2.putText(img, send_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        
        y += 40
        # Main stats
        cv2.putText(img, f'Total: {stats["total"]}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        cv2.putText(img, f'OK: {stats["ok"]}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 30
        cv2.putText(img, f'DEFECTIVE: {stats["defective"]}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        y += 25
        cv2.putText(img, f'Holes (counted): {stats["total_holes"]}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        y += 40
        # Current tracking
        if stats['current_id']:
            cv2.putText(img, f'Tracking: Product #{stats["current_id"]}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        else:
            cv2.putText(img, 'Tracking: None', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
        
        # Recent products
        y += 40
        cv2.putText(img, 'RECENT:', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y += 20
        
        recent = tracker.completed_products[-5:] if tracker.completed_products else []
        for product in reversed(recent):
            color = (0, 255, 0) if not product.is_defective() else (0, 0, 255)
            text = f'#{product.id}: {product.get_status()}'
            cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y += 18
        
        return img
    
    def display_frames(self, main_frame, stats_frame):
        # Scale main frame
        scaled_main = cv2.resize(main_frame, (self.display_width, self.display_height))
        scaled_stats = cv2.resize(stats_frame, (600, 1050))
        
        cv2.imshow('CountifyTech Inspector', scaled_main)
        cv2.imshow('Statistics', scaled_stats)

# ===== MAIN APPLICATION =====
def run_inspection(model_path='/app/model8.engine', vercel_url="https://your-backend-url.vercel.app"):
    # Initialize components
    backend = BackendSender(vercel_url)
    tracker = ProductTracker(backend)
    camera_controller = CameraController()
    processor = DetectionProcessor(model_path)
    display = DisplayManager()
    
    frame_interval = 1.0 / Config.TARGET_FPS
    
    try:
        while True:
            start_time = time.time()
            
            # Update camera controller and check if detection is ready
            can_detect = camera_controller.update()
            
            # Read frames
            frame, camera_name, is_camera_0 = camera_controller.read_frames()
            if frame is None:
                print("Camera read failed")
                break
            
            # Process detections only if ready
            detections = []
            results = None
            if can_detect:
                detections, results = processor.process_frame(frame, is_camera_0)
            
            # Check for product in ROI
            product_in_roi = False
            if detections:
                product_in_roi = processor.has_product_in_roi(detections, is_camera_0)
            
            # Update confirmation tracker
            confirmed = False
            if can_detect:
                confirmed = camera_controller.update_confirmation(product_in_roi, is_camera_0)
            
            # Main logic with confirmation
            if is_camera_0 and can_detect and not tracker.current_product:
                # Camera 0: Look for new products with confirmation
                if confirmed:
                    tracker.start_new_product()
                    camera_controller.request_switch()
                    camera_controller.reset_confirmation(is_camera_0)
            
            elif not is_camera_0 and can_detect and tracker.current_product:
                # Camera 1: Look for product exit with confirmation
                if confirmed:
                    tracker.finish_current_product()
                    camera_controller.request_switch()
                    camera_controller.reset_confirmation(is_camera_0)
            
            # Update defect counts only when we have a current product
            if tracker.current_product and detections:
                holes, dents = processor.count_defects(detections, is_camera_0)
                if is_camera_0:
                    tracker.update_from_top_camera(holes, dents)
                else:
                    tracker.update_from_side_camera(holes, dents)
            
            # Draw everything
            display.draw_detections(frame, detections, results)
            display.draw_roi_line(frame, can_detect)
            display.add_camera_label_with_confirmation(frame, camera_name, camera_controller)
            
            # Create and display windows
            stats_frame = display.create_stats_window(tracker, camera_controller, backend)
            display.display_frames(frame, stats_frame)
            
            # Exit check
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Frame rate control
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        print("?? Shutting down...")
        backend.shutdown()
        camera_controller.cleanup()
        cv2.destroyAllWindows()
        print("? Shutdown complete")

if __name__ == '__main__':
    # Configure your backend URL using environment variable or default
    # Set environment variable: export BACKEND_URL="https://your-backend-url.vercel.app" (Linux/Mac)
    # Or: $env:BACKEND_URL="https://your-backend-url.vercel.app" (Windows PowerShell)
    BACKEND_URL = os.getenv('BACKEND_URL', 'https://your-backend-url.vercel.app')
    
    run_inspection(vercel_url=BACKEND_URL)