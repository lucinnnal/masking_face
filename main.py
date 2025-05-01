from Utils import VideoProcessor, modify_json_paths

def main():
    """
    메인 실행 함수
    사용 예시:
    1. JSON 파일 경로를 초기화 시점에 전달:
        python utils.py --json_path video_paths.json
    2. 개별 비디오 처리:
        python utils.py --video_path input.mp4 --output_path output.mp4
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Processing Tool')
    parser.add_argument('--json_path', type=str, help='JSON file containing video paths')
    parser.add_argument('--video_path', type=str, help='Single video file path')
    parser.add_argument('--output_path', type=str, help='Output path for single video')
    args = parser.parse_args()
    
    # VideoProcessor 인스턴스 생성
    processor = VideoProcessor(
        json_path=args.json_path,
        yolo_confidence=0.3,
        yolo_iou=0.45,
        face_threshold=0.2,
        mask_scale=1.3,
        blur_type='blur',
        ellipse=True
    )
    
    if args.json_path:
        # Multiple videos
        processed_videos = processor.process_videos_from_json(args.json_path, args.output_path)
        print("\nProcessed video files:")
        for video_path in processed_videos:
            print(f"- {video_path}")
    
    elif args.video_path and args.output_path:
        # Single video
        processor.process_video(args.video_path, args.output_path)
        print(f"\nProcessed video saved to: {args.output_path}")
    
    else:
        print("Please provide either a JSON file path or both input and output video paths.")

if __name__ == "__main__":
    main()