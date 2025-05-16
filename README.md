** Cropping & Masking 설명 **


** 가상환경 설정 **
conda env create -f urp2025.yml


** 실행 방법 **
1. 여러 개의 비디오를 한번에 처리 : json 파일에 데이터 경로들을 저장(리스트로) 되어있을 때 json_path랑 ouput_path 지정해주고 실행
EX)
python main.py --json_path video_paths.json --output_path ./processed_video --process_type crop


2. 하나의 비디오만 받아서 처리하기
EX)
python main.py --video_path /Users/kipyokim/Desktop/data_preprocessing/HEROES/BOREDOM/01/01_B3_B.mp4 --output_path output.mp4


# JSON 파일로 크롭+마스킹 처리
python main.py --json_path video_paths.json --output-path ./processed --process_type full

# JSON 파일로 크롭만 처리
python main.py --json_path video_paths.json --output_path ./cropped_only --process_type crop

# 단일 비디오 크롭+마스킹
python main.py --video_path input.mp4 --output_path output.mp4 --process_type full

# 단일 비디오 크롭만
python main.py --video_path input.mp4 --output_path output.mp4 --process_type crop