import cv2
import imageio
import imageio.v2 as iio
import numpy as np
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

if __name__ == "__main__":
    # VideoAnonymizer 인스턴스 생성
    anonymizer = Facemasker(
        threshold=0.2,
        mask_scale=1.3,
        replacewith='blur',  # 'blur', 'solid', 'mosaic', 'img' 중 선택
        ellipse=True,
        draw_scores=False,
        mosaicsize=20
    )
    
    # 비디오 처리 예시
    input_video = "./sample/01_B1_A 복사본.mp4"
    output_video = "./output.mp4"
    anonymizer.process_video(input_video, output_video)