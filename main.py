import torch
import cv2
import numpy as np
from pathlib import Path

# YOLOv5 모델 로드 및 설정
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
if torch.cuda.is_available():
    model.cuda()
model.conf = 0.3  # confidence threshold 낮춤
model.iou = 0.45  # NMS IoU threshold 조정

def process_video(video_path, output_path):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    # 버퍼 설정
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    # 비디오 속성 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 결과 비디오 저장을 위한 VideoWriter 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 크롭된 비디오를 위한 고정 크기 설정 (예: 256x256)
    crop_path = str(Path(output_path) / 'cropped_video2.mp4')
    
    # 사람 추적을 위한 딕셔너리 추가
    tracking_dict = {}  # {id: {'count': int, 'position': (x1,y1,x2,y2)}}
    next_id = 0
    
    # 첫번째 패스: 최대 바운딩 박스 크기 찾기
    max_width = 0
    max_height = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame)
        persons = results.xyxy[0][results.xyxy[0][:, -1] == 0]
        
        for bbox in persons:
            x1, y1, x2, y2 = map(int, bbox[:4])
            width = x2 - x1
            height = y2 - y1
            max_width = max(max_width, width)
            max_height = max(max_height, height)
    
    # 비디오 포인터 리셋
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 크롭 영역 크기 계산 (50% 확장)
    crop_size = int(max(max_width, max_height) * 1.5)
    
    # VideoWriter 수정 (정사각형 출력)
    crop_writer = cv2.VideoWriter(
        crop_path,
        fourcc,
        fps,
        (crop_size, crop_size)
    )
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # GPU 사용 가능시 GPU로 처리
        if torch.cuda.is_available():
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame, size=640)  # 입력 크기 고정
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            results = model(frame)
        
        # 현재 프레임의 바운딩 박스
        current_boxes = results.xyxy[0][results.xyxy[0][:, -1] == 0]
        
        # 현재 프레임에서 감지된 사람들 처리
        current_tracked = set()
        
        for bbox in current_boxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # 가장 가까운 이전 트래킹 찾기
            min_dist = float('inf')
            matched_id = None
            
            for track_id, track_info in tracking_dict.items():
                prev_box = track_info['position']
                prev_center = ((prev_box[0] + prev_box[2]) // 2, 
                             (prev_box[1] + prev_box[3]) // 2)
                dist = np.sqrt((current_center[0] - prev_center[0])**2 + 
                             (current_center[1] - prev_center[1])**2)
                
                if dist < min_dist and dist < 100:  # 100픽셀 임계값
                    min_dist = dist
                    matched_id = track_id
            
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
        for track_id in list(tracking_dict.keys()):
            if track_id not in current_tracked:
                del tracking_dict[track_id]
        
        if tracking_dict:
            # 가장 오래 추적된 사람 선택
            longest_tracked_id = max(tracking_dict.keys(), 
                                   key=lambda k: tracking_dict[k]['count'])
            target_box = tracking_dict[longest_tracked_id]['position']
            x1, y1, x2, y2 = target_box
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 크롭용 중심점 계산
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 고정된 크기로 크롭 영역 계산
            half_size = crop_size // 2
            crop_x1 = max(0, center_x - half_size)
            crop_y1 = max(0, center_y - half_size)
            crop_x2 = min(frame_width, center_x + half_size)
            crop_y2 = min(frame_height, center_y + half_size)
            
            # 크롭 및 리사이즈
            cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop_x2-crop_x1 != crop_size or crop_y2-crop_y1 != crop_size:
                padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                y_offset = max(0, -center_y + half_size)
                x_offset = max(0, -center_x + half_size)
                padded[y_offset:y_offset+cropped.shape[0], 
                       x_offset:x_offset+cropped.shape[1]] = cropped
                cropped = padded
            
            # 크롭된 영상 저장
            crop_writer.write(cropped)
        
        # 결과 프레임 저장
        out.write(frame)
        
        # 화면에 결과 표시 (선택사항)
        """"
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        """
    
    # 리소스 해제
    cap.release()
    out.release()
    crop_writer.release()
    cv2.destroyAllWindows()

# 사용 예시
video_path = './sample/01_B1_A 복사본.mp4'  # 입력 비디오 경로
output_path = './cropped/'  # 출력 비디오 경로
process_video(video_path, output_path)