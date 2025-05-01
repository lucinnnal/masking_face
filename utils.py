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

# Face masker -> Blurring face
class Facemasker:
    def __init__(self, 
                 threshold=0.2,
                 mask_scale=1.3,
                 replacewith='blur',
                 ellipse=True,
                 draw_scores=False,
                 mosaicsize=20,
                 backend='auto'):
        """
        비디오/이미지 익명화를 위한 클래스
        Args:
            threshold (float): 얼굴 검출 임계값
            mask_scale (float): 마스크 크기 스케일
            replacewith (str): 'blur', 'solid', 'none', 'img', 'mosaic' 중 선택
            ellipse (bool): 타원형 마스크 사용 여부
            draw_scores (bool): 검출 점수 표시 여부
            mosaicsize (int): 모자이크 크기
            backend (str): 'auto', 'onnxrt', 'opencv' 중 선택
        """
        self.threshold = threshold
        self.mask_scale = mask_scale
        self.replacewith = replacewith
        self.ellipse = ellipse
        self.draw_scores = draw_scores
        self.mosaicsize = mosaicsize
        
        # CenterFace 모델 초기화
        self.centerface = CenterFace(in_shape=None, backend=backend)
        
        # 대체 이미지 초기화
        self.replaceimg = None

    def set_replace_image(self, image_path):
        """대체 이미지 설정"""
        if self.replacewith == 'img':
            self.replaceimg = imageio.imread(image_path)
    
    def _draw_anonymization(self, frame, dets):
        """프레임에 익명화 적용"""
        for det in dets:
            boxes, score = det[:4], det[4]
            x1, y1, x2, y2 = boxes.astype(int)
            x1, y1, x2, y2 = self._scale_bb(x1, y1, x2, y2)
            
            # 프레임 경계 확인
            y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
            x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)
            
            self._apply_anonymization(frame, score, x1, y1, x2, y2)
        
        return frame

    def _scale_bb(self, x1, y1, x2, y2):
        """바운딩 박스 크기 조정"""
        s = self.mask_scale - 1.0
        h, w = y2 - y1, x2 - x1
        y1 -= h * s
        y2 += h * s
        x1 -= w * s
        x2 += w * s
        return np.round([x1, y1, x2, y2]).astype(int)

    def _apply_anonymization(self, frame, score, x1, y1, x2, y2):
        """익명화 효과 적용"""
        if self.replacewith == 'solid':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
            
        elif self.replacewith == 'blur':
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
                
        elif self.replacewith == 'mosaic':
            for y in range(y1, y2, self.mosaicsize):
                for x in range(x1, x2, self.mosaicsize):
                    pt1 = (x, y)
                    pt2 = (min(x2, x + self.mosaicsize - 1), min(y2, y + self.mosaicsize - 1))
                    color = tuple(map(int, frame[y, x]))
                    cv2.rectangle(frame, pt1, pt2, color, -1)
                    
        if self.draw_scores:
            cv2.putText(
                frame, f'{score:.2f}', (x1, y1 - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0)
            )

    def _get_ellipse_coords(self, height, width):
        """타원형 좌표 계산"""
        from skimage.draw import ellipse
        return ellipse(height // 2, width // 2, height // 2, width // 2)

    def process_video(self, input_path, output_path):
        """
        비디오 처리
        Args:
            input_path (str): 입력 비디오 경로
            output_path (str): 출력 비디오 경로
        """
        # 출력 디렉토리 생성
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 비디오 읽기
        cap = cv2.VideoCapture(input_path)
        
        # 비디오 속성
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 출력 비디오 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 얼굴 검출
            dets, _ = self.centerface(frame, threshold=self.threshold)
            
            # 익명화 적용
            processed_frame = self._draw_anonymization(frame, dets)
            
            # 프레임 저장
            out.write(processed_frame)
        
        # 리소스 해제
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def process_image(self, input_path, output_path):
        """
        이미지 처리
        Args:
            input_path (str): 입력 이미지 경로
            output_path (str): 출력 이미지 경로
        """
        # 이미지 읽기
        frame = iio.imread(input_path)
        
        # 얼굴 검출
        dets, _ = self.centerface(frame, threshold=self.threshold)
        
        # 익명화 적용
        processed_frame = self._draw_anonymization(frame, dets)
        
        # 이미지 저장
        imageio.imsave(output_path, processed_frame)

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

# VideoProcessor : Cropper + Masker
class VideoProcessor:
    def __init__(self,
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

    def process_video(self, video_path, output_path):
        """
        비디오를 크롭하고 얼굴을 마스킹하여 저장
        Args:
            video_path (str): 입력 비디오 경로
            output_path (str): 출력 비디오 경로
        """
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

if __name__ == "__main__":

    processor = VideoProcessor(
        yolo_confidence=0.3,
        yolo_iou=0.45,
        face_threshold=0.2,
        mask_scale=1.3,
        blur_type='blur',
        ellipse=True
    )

    video_path = './sample/04_B2_A 복사본.mp4'
    output_path = '04_B2_A 복사본_final_processed.mp4'
    
    processor.process_video(video_path, output_path)
    print(f"Processing completed: {output_path}")