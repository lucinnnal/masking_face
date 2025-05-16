from Utils import VideoProcessor, VideoCropper, modify_json_paths

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Processing Tool')
    parser.add_argument('--json_path', type=str, help='JSON file containing video paths')
    parser.add_argument('--video_path', type=str, help='Single video file path')
    parser.add_argument('--output_path', type=str, help='Output path for single video')
    parser.add_argument('--process_type', type=str, choices=['crop', 'full'], 
                       default='full', help='Processing type: crop only or crop+masking')
    args = parser.parse_args()
    
    # 처리 유형에 따라 적절한 프로세서 생성
    if args.process_type == 'full':
        processor = VideoProcessor(
            json_path=args.json_path,
            yolo_confidence=0.3,
            yolo_iou=0.45,
            face_threshold=0.2,
            mask_scale=1.3,
            blur_type='blur',
            ellipse=True
        )
    else:  # crop only
        processor = VideoCropper(
            json_path=args.json_path,
            yolo_confidence=0.3,
            yolo_iou=0.45
        )
    
    if args.json_path:
        # Multiple videos
        processed_videos = processor.process_videos_from_json(args.json_path)
        print(f"\nProcessed video files ({args.process_type} processing):")
        for video_path in processed_videos:
            print(f"- {video_path}")
    
    elif args.video_path and args.output_path:
        # Single video
        processor.process_video(args.video_path, args.output_path)
        print(f"\nProcessed video ({args.process_type} processing) saved to: {args.output_path}")
    
    else:
        print("Please provide either a JSON file path or both input and output video paths.")

if __name__ == "__main__":
    main()