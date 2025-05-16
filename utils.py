import torch
import numpy as np
import cv2
import numpy as np
from pathlib import Path
import imageio
import imageio.v2 as iio
from src.centerface import CenterFace
from pathlib import Path
import os
import json

# Modify json path
def modify_json_paths(json_path):
    with open(json_path, 'r') as f:
        video_paths = json.load(f)
    
    modified_paths = [path.replace('../', './') for path in video_paths]
    
    with open(json_path, 'w') as f:
        json.dump(modified_paths, f, indent=4)
    
    print(f"Modified {len(video_paths)} paths in {json_path}")

# VideoProcessor : Cropper + Masker
class VideoProcessor:
    def __init__(self,
                 json_path=None,
                 yolo_confidence=0.3,
                 yolo_iou=0.45,
                 face_threshold=0.2,
                 mask_scale=1.3,
                 blur_type='blur',
                 ellipse=True,
                 draw_scores=False,
                 mosaicsize=20):
        """
        비디오 크롭 및 얼굴 마스킹을 위한 통합 클래스
        Args:
            json_path (str, optional): 비디오 경로 리스트가 담긴 JSON 파일 경로
            yolo_confidence (float): YOLOv5 confidence threshold
            yolo_iou (float): YOLOv5 IOU threshold
            face_threshold (float): 얼굴 검출 임계값
            mask_scale (float): 마스크 크기 스케일
            blur_type (str): 'blur' or 'mosaic'
            ellipse (bool): 타원형 마스크 사용 여부
            draw_scores (bool): 검출 점수 표시 여부
            mosaicsize (int): 모자이크 크기
        """
        # YOLO 모델 초기화
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        if torch.cuda.is_available():
            self.yolo_model.cuda()
        self.yolo_model.conf = yolo_confidence
        self.yolo_model.iou = yolo_iou
        
        # CenterFace 모델 초기화
        self.face_detector = CenterFace(backend='auto')
        self.face_threshold = face_threshold
        self.mask_scale = mask_scale
        self.blur_type = blur_type
        self.ellipse = ellipse
        self.draw_scores = draw_scores
        self.mosaicsize = mosaicsize
        
        # JSON 파일 경로 저장
        self.json_path = json_path

    def process_video(self, video_path, output_dir):
        """
        비디오를 크롭하고 얼굴을 마스킹하여 저장
        Args:
            video_path (str): 입력 비디오 경로
            output_path (str): 출력 비디오 경로
        """
        # 비디오 파일 존재 확인
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        # 출력 디렉토리 확인 및 생성
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        output_path = str(output_dir / f"{video_name}_processed.mp4")

        cap = cv2.VideoCapture(video_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 최대 바운딩 박스 크기 계산
        max_size = self._get_max_bbox_size(cap)
        crop_size = int(max_size * 1.5)
        
        # 출력 비디오 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_size, crop_size))
        
        tracking_dict = {}
        next_id = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. 사람 검출 및 크롭
            if torch.cuda.is_available():
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.yolo_model(frame_rgb, size=640)
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                results = self.yolo_model(frame)
            
            current_boxes = results.xyxy[0][results.xyxy[0][:, -1] == 0]
            tracking_dict = self._update_tracking(current_boxes, tracking_dict, next_id)
            
            if tracking_dict:
                cropped = self._get_cropped_frame(frame, tracking_dict, crop_size)
                
                # 2. 얼굴 검출 및 마스킹
                dets, _ = self.face_detector(cropped, threshold=self.face_threshold)
                masked_frame = self._apply_face_masking(cropped.copy(), dets)
                
                # 프레임 저장
                out.write(masked_frame)
        
        # 리소스 해제
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def _apply_face_masking(self, frame, dets):
        """얼굴 영역 마스킹 적용"""
        for det in dets:
            boxes, score = det[:4], det[4]
            x1, y1, x2, y2 = boxes.astype(int)
            
            if self.blur_type == 'blur':
                bf = 2
                blurred_box = cv2.blur(
                    frame[y1:y2, x1:x2],
                    (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
                )
                if self.ellipse:
                    roibox = frame[y1:y2, x1:x2]
                    ey, ex = self._get_ellipse_coords((y2 - y1), (x2 - x1))
                    roibox[ey, ex] = blurred_box[ey, ex]
                    frame[y1:y2, x1:x2] = roibox
                else:
                    frame[y1:y2, x1:x2] = blurred_box
                    
            elif self.blur_type == 'mosaic':
                for y in range(y1, y2, self.mosaicsize):
                    for x in range(x1, x2, self.mosaicsize):
                        pt1 = (x, y)
                        pt2 = (min(x2, x + self.mosaicsize - 1), 
                              min(y2, y + self.mosaicsize - 1))
                        color = tuple(map(int, frame[y, x]))
                        cv2.rectangle(frame, pt1, pt2, color, -1)
                        
            if self.draw_scores:
                cv2.putText(frame, f'{score:.2f}', (x1, y1 - 20),
                           cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
        
        return frame

    def _get_max_bbox_size(self, cap):
        """최대 바운딩 박스 크기 계산"""
        max_size = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            results = self.yolo_model(frame)
            persons = results.xyxy[0][results.xyxy[0][:, -1] == 0]
        
            for bbox in persons:
                x1, y1, x2, y2 = map(int, bbox[:4])
                max_size = max(max_size, x2 - x1, y2 - y1)
    
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return max_size

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

    def _get_ellipse_coords(self, height, width):
        """타원형 좌표 계산"""
        from skimage.draw import ellipse
        return ellipse(height // 2, width // 2, height // 2, width // 2)

    def process_videos_from_json(self, json_path, output_dir='processed', batch_size=2):
        """
        JSON 파일에 있는 비디오 파일들을 배치 단위로 처리
        Args:
            json_path (str): 비디오 경로 리스트가 담긴 JSON 파일 경로
            output_dir (str): 처리된 비디오들을 저장할 디렉토리 경로
            batch_size (int): 한 번에 처리할 비디오 수
        Returns:
            list: 처리된 비디오 파일 경로 리스트
        """
        # JSON 파일 읽기
        with open(json_path, 'r') as f:
            video_paths = json.load(f)
            
        if not isinstance(video_paths, list):
            raise ValueError("JSON 파일은 비디오 경로 리스트를 포함해야 합니다.")
            
        # 출력 디렉토리 생성
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_paths = []
        total_videos = len(video_paths)
        
        # 배치 단위로 처리
        for batch_idx in range(0, total_videos, batch_size):
            batch_videos = video_paths[batch_idx:batch_idx + batch_size]
            print(f"\nProcessing batch {batch_idx//batch_size + 1}/{(total_videos+batch_size-1)//batch_size}")
            
            # 배치 내 각 비디오 처리
            for idx, video_path in enumerate(batch_videos, 1):
                try:
                    video_name = Path(video_path).stem
                    output_path = str(output_dir / f"{video_name}_processed.mp4")
                    
                    print(f"Processing video {batch_idx + idx}/{total_videos}: {video_name}")
                    self.process_video(video_path, output_path)
                    
                    processed_paths.append(output_path)
                    print(f"Completed: {output_path}")
                    
                except Exception as e:
                    print(f"Error processing {video_path}: {str(e)}")
                    continue
            
            # 배치 처리 후 메모리 정리 및 모델 리셋
            print(f"Batch {batch_idx//batch_size + 1} completed. Resetting models...")
            
            # 메모리 정리
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # YOLO 모델 리셋
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            if torch.cuda.is_available():
                self.yolo_model.cuda()
            self.yolo_model.conf = self.yolo_model.conf
            self.yolo_model.iou = self.yolo_model.iou
            
            # CenterFace 모델 리셋
            self.face_detector = CenterFace(backend='auto')
            
            print("Models reset completed. Taking a short break...")
            import time
            time.sleep(5)  # 5초 대기
        
        print(f"\nAll processing completed. {len(processed_paths)}/{total_videos} videos processed successfully.")
        return processed_paths

class VideoCropper:
    def __init__(self,
                 json_path=None,
                 yolo_confidence=0.3,
                 yolo_iou=0.45):
        """
        비디오 크롭을 위한 클래스
        Args:
            json_path (str, optional): 비디오 경로 리스트가 담긴 JSON 파일 경로
            yolo_confidence (float): YOLOv5 confidence threshold
            yolo_iou (float): YOLOv5 IOU threshold
        """
        # YOLO 모델 초기화
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        if torch.cuda.is_available():
            self.yolo_model.cuda()
        self.yolo_model.conf = yolo_confidence
        self.yolo_model.iou = yolo_iou
        
        # JSON 파일 경로 저장
        self.json_path = json_path

    def process_video(self, video_path, output_dir):
        """
        비디오를 크롭하여 저장
        Args:
            video_path (str): 입력 비디오 경로
            output_path (str): 출력 비디오 경로
        """
        # 비디오 파일 존재 확인
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        # 출력 디렉토리 확인 및 생성
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        output_path = str(output_dir / f"{video_name}_cropped.mp4")

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 최대 바운딩 박스 크기 계산
        max_size = self._get_max_bbox_size(cap)
        crop_size = int(max_size * 1.5)
        
        # 출력 비디오 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_size, crop_size))
        
        tracking_dict = {}
        next_id = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if torch.cuda.is_available():
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.yolo_model(frame_rgb, size=640)
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                results = self.yolo_model(frame)
            
            current_boxes = results.xyxy[0][results.xyxy[0][:, -1] == 0]
            tracking_dict = self._update_tracking(current_boxes, tracking_dict, next_id)
            
            if tracking_dict:
                cropped = self._get_cropped_frame(frame, tracking_dict, crop_size)
                out.write(cropped)
        
        # 리소스 해제
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def process_videos_from_json(self, json_path, output_dir='cropped', batch_size=2):
        """
        JSON 파일에 있는 비디오 파일들을 배치 단위로 처리
        Args:
            json_path (str): 비디오 경로 리스트가 담긴 JSON 파일 경로
            output_dir (str): 처리된 비디오들을 저장할 디렉토리 경로
            batch_size (int): 한 번에 처리할 비디오 수
        Returns:
            list: 처리된 비디오 파일 경로 리스트
        """
        # JSON 파일 읽기
        with open(json_path, 'r') as f:
            video_paths = json.load(f)
            
        if not isinstance(video_paths, list):
            raise ValueError("JSON 파일은 비디오 경로 리스트를 포함해야 합니다.")
            
        # 출력 디렉토리 생성
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_paths = []
        total_videos = len(video_paths)
        
        # 배치 단위로 처리
        for batch_idx in range(0, total_videos, batch_size):
            batch_videos = video_paths[batch_idx:batch_idx + batch_size]
            print(f"\nProcessing batch {batch_idx//batch_size + 1}/{(total_videos+batch_size-1)//batch_size}")
            
            # 배치 내 각 비디오 처리
            for idx, video_path in enumerate(batch_videos, 1):
                try:
                    video_name = Path(video_path).stem
                    output_path = str(output_dir / f"{video_name}_cropped.mp4")
                    
                    print(f"Processing video {batch_idx + idx}/{total_videos}: {video_name}")
                    self.process_video(video_path, output_path)
                    
                    processed_paths.append(output_path)
                    print(f"Completed: {output_path}")
                    
                except Exception as e:
                    print(f"Error processing {video_path}: {str(e)}")
                    continue
            
            # 배치 처리 후 메모리 정리 및 모델 리셋
            print(f"Batch {batch_idx//batch_size + 1} completed. Resetting models...")
            
            # 메모리 정리
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # YOLO 모델 리셋
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            if torch.cuda.is_available():
                self.yolo_model.cuda()
            self.yolo_model.conf = self.yolo_model.conf
            self.yolo_model.iou = self.yolo_model.iou
            
            print("Models reset completed. Taking a short break...")
            import time
            time.sleep(5)  # 5초 대기
        
        print(f"\nAll processing completed. {len(processed_paths)}/{total_videos} videos processed successfully.")
        return processed_paths

    # Helper methods (VideoProcessor와 동일)
    _get_max_bbox_size = VideoProcessor._get_max_bbox_size
    _update_tracking = VideoProcessor._update_tracking
    _find_closest_track = VideoProcessor._find_closest_track
    _get_cropped_frame = VideoProcessor._get_cropped_frame