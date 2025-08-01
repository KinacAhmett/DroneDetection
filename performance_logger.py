import os
import cv2
import json
from datetime import datetime

class PerformanceLogger:
    def __init__(self, test_name):
        self.test_name = test_name
        self.results = {
            'test_name': test_name,
            'timestamp': datetime.now().isoformat(),
            'detections': [],
            'tracking_success': 0,
            'false_positives': 0,
            'center_hits': 0,
            'total_frames': 0
        }
    
    def log_detection(self, frame_num, bbox, is_center_focused, is_tracking):
        """Her tespit için log"""
        detection = {
            'frame': frame_num,
            'bbox': bbox,
            'center_distance': self.calculate_center_distance(bbox),
            'is_center_focused': is_center_focused,
            'is_tracking': is_tracking
        }
        self.results['detections'].append(detection)
        
        if is_center_focused:
            self.results['center_hits'] += 1
        
        if is_tracking:
            self.results['tracking_success'] += 1
    
    def calculate_center_distance(self, bbox):
        """Merkez mesafesi hesapla"""
        center_x, center_y = 320, 256
        drone_x = bbox[0] + bbox[2]/2
        drone_y = bbox[1] + bbox[3]/2
        return ((drone_x - center_x)**2 + (drone_y - center_y)**2)**0.5
    
    def save_results(self):
        """Sonuçları kaydet"""
        filename = f"performance_{self.test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Summary yazdır
        total_detections = len(self.results['detections'])
        if total_detections > 0:
            center_ratio = self.results['center_hits'] / total_detections * 100
            tracking_ratio = self.results['tracking_success'] / total_detections * 100
            
            print(f"\n=== {self.test_name} SONUÇLARI ===")
            print(f"Toplam Tespit: {total_detections}")
            print(f"Merkez Odaklı: {self.results['center_hits']} (%{center_ratio:.1f})")
            print(f"Tracking Başarı: {self.results['tracking_success']} (%{tracking_ratio:.1f})")
            print(f"Ortalama Merkez Mesafesi: {sum(d['center_distance'] for d in self.results['detections'])/total_detections:.1f}")
            print(f"Sonuçlar kaydedildi: {filename}")

# Test videoları için performance logger kullanımı
if __name__ == "__main__":
    # Örnek kullanım:
    logger = PerformanceLogger("baseline_subset36_run22")
    
    # Her tespit için:
    # logger.log_detection(frame_num, bbox, is_center_focused, is_tracking)
    
    logger.save_results()
