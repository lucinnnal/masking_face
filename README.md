** Cropping & Masking 설명 **


** 가상환경 설정 **
conda env create -f urp2025.yml


** 실행 방법 **
1. 여러 개의 비디오를 한번에 처리 : json 파일에 데이터 경로들을 저장(리스트로) 되어있을 때 json_path랑 ouput_path 지정해주고 실행
EX)
python main.py --json_path video_paths.json --output_path ./processed_video


2. 하나의 비디오만 받아서 처리하기
EX)
python main.py --video_path /Users/kipyokim/Desktop/data_preprocessing/HEROES/BOREDOM/01/01_B3_B.mp4 --output_path output.mp4