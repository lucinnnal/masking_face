import torch
import cv2
import numpy as np
from pathlib import Path

# Cropping video after detecting human
class VideoCropper:
    def __init__(self, confidence=0.3, iou=0.45):
        """
        비디오 처리를 위한 클래스 초기화
        Args:
            confidence (float): YOLOv5 모델의 confidence threshold
            iou (float): YOLOv5 모델의 NMS IoU threshold
        """
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.conf = confidence
        self.model.iou = iou
        
    def process_video(self, video_path, output_path):
        """
        비디오를 처리하고 크롭된 결과만 저장
        Args:
            video_path (str): 입력 비디오 경로
            output_path (str): 출력 디렉토리 경로
        Returns:
            str: 크롭된 비디오 저장 경로
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        # 비디오 속성 가져오기
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 출력 경로 설정
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        crop_output = str(output_dir / f'cropped_{Path(video_path).stem}.mp4')
        
        # 최대 바운딩 박스 크기 계산
        max_size = self._get_max_bbox_size(cap)
        crop_size = int(max_size * 1.5)
        
        # 크롭 비디오 writer 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        crop_writer = cv2.VideoWriter(crop_output, fourcc, fps, (crop_size, crop_size))
        
        # 메인 처리
        tracking_dict = {}
        next_id = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if torch.cuda.is_available():
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model(frame, size=640)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                results = self.model(frame)
            
            current_boxes = results.xyxy[0][results.xyxy[0][:, -1] == 0]
            tracking_dict = self._update_tracking(current_boxes, tracking_dict, next_id)
            
            if tracking_dict:
                cropped = self._get_cropped_frame(frame, tracking_dict, crop_size)
                crop_writer.write(cropped)
        
        # 리소스 해제
        cap.release()
        crop_writer.release()
        cv2.destroyAllWindows()
        
        return crop_output
    
    def _get_max_bbox_size(self, cap):
        """최대 바운딩 박스 크기 계산"""
        max_size = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model(frame)
            persons = results.xyxy[0][results.xyxy[0][:, -1] == 0]
            
            for bbox in persons:
                x1, y1, x2, y2 = map(int, bbox[:4])
                max_size = max(max_size, x2 - x1, y2 - y1)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return max_size
    
    def _process_frames(self, cap, out, crop_writer, crop_size):
        """프레임 처리 및 트래킹"""
        tracking_dict = {}
        next_id = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if torch.cuda.is_available():
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model(frame, size=640)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                results = self.model(frame)
            
            current_boxes = results.xyxy[0][results.xyxy[0][:, -1] == 0]
            tracking_dict = self._update_tracking(current_boxes, tracking_dict, next_id)
            
            if tracking_dict:
                cropped = self._get_cropped_frame(frame, tracking_dict, crop_size)
                crop_writer.write(cropped)
            
            out.write(frame)

    def _update_tracking(self, current_boxes, tracking_dict, next_id):
        """트래킹 정보 업데이트"""
        current_tracked = set()
        
        for bbox in current_boxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            matched_id = self._find_closest_track(current_center, tracking_dict)
            
            if matched_id is not None:
                tracking_dict[matched_id]['count'] += 1
                tracking_dict[matched_id]['position'] = (x1, y1, x2, y2)
                current_tracked.add(matched_id)
            else:
                tracking_dict[next_id] = {
                    'count': 1,
                    'position': (x1, y1, x2, y2)
                }
                current_tracked.add(next_id)
                next_id += 1
        
        # 미검출된 트래킹 제거
        return {k: v for k, v in tracking_dict.items() if k in current_tracked}
    
    def _find_closest_track(self, current_center, tracking_dict):
        """가장 가까운 트래킹 찾기"""
        min_dist = float('inf')
        matched_id = None
        
        for track_id, track_info in tracking_dict.items():
            prev_box = track_info['position']
            prev_center = ((prev_box[0] + prev_box[2]) // 2, 
                         (prev_box[1] + prev_box[3]) // 2)
            dist = np.sqrt((current_center[0] - prev_center[0])**2 + 
                         (current_center[1] - prev_center[1])**2)
            
            if dist < min_dist and dist < 100:
                min_dist = dist
                matched_id = track_id
        
        return matched_id

    def _get_cropped_frame(self, frame, tracking_dict, crop_size):
        """크롭된 프레임 생성"""
        longest_tracked_id = max(tracking_dict.keys(), 
                               key=lambda k: tracking_dict[k]['count'])
        target_box = tracking_dict[longest_tracked_id]['position']
        x1, y1, x2, y2 = target_box
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        half_size = crop_size // 2
        crop_x1 = max(0, center_x - half_size)
        crop_y1 = max(0, center_y - half_size)
        crop_x2 = min(frame.shape[1], center_x + half_size)
        crop_y2 = min(frame.shape[0], center_y + half_size)
        
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_x2-crop_x1 != crop_size or crop_y2-crop_y1 != crop_size:
            padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            y_offset = max(0, -center_y + half_size)
            x_offset = max(0, -center_x + half_size)
            padded[y_offset:y_offset+cropped.shape[0], 
                   x_offset:x_offset+cropped.shape[1]] = cropped
            cropped = padded
            
        return cropped

if __name__ == "__main__":
    processor = VideoCropper(confidence=0.3, iou=0.45)
    
    video_path = './sample/01_B1_A 복사본.mp4'
    output_path = './cropped/'
    
    crop_output = processor.process_video(video_path, output_path)
    print(f"Cropped video saved: {crop_output}")