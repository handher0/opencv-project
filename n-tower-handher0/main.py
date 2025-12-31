import argparse
import sys
import os


# ultralytics를 불러올 때 나오는 로그 제거
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# 로그 차단 시작
with SuppressOutput():
    import cv2
    import numpy as np
    from ultralytics import YOLO


# 로그 차단 해제

def parse_args():
    p = argparse.ArgumentParser(description="N Seoul Tower Detector")
    p.add_argument("--input", required=True, type=str, help="path to input image")
    p.add_argument("--task", required=True, type=str, choices=["presence", "bbox"], help="Task type")
    # [추가] 디버깅 모드 플래그 (이 옵션을 쓸 때만 이미지를 띄움)
    p.add_argument("--debug", action="store_true", help="Enable debug visualization")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. 이미지 로드
    img = cv2.imread(args.input)
    if img is None:
        sys.exit(1)

    # 2. 모델 로드
    try:
        with SuppressOutput():
            model = YOLO('best.pt')
    except Exception:
        sys.exit(1)

    # 3. 추론
    results = model(img, verbose=False, conf=0.51)

    best_box = None
    if results[0].boxes:
        best_box = results[0].boxes[0]

    task = args.task

    # 4. 출력 및 디버깅
    if task == "presence":
        if best_box is not None:
            print("true")
        else:
            print("false")

    elif task == "bbox":
        if best_box is not None:
            # 좌표 계산
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            x = int(round(x1))
            y = int(round(y1))
            w = int(round(x2 - x1))
            h = int(round(y2 - y1))

            # 정답 출력
            print(f"{x},{y},{w},{h}")

            # --debug 옵션이 있을 때만 실행
            if args.debug:
                # 1. 빨간색 사각형 그리기
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # 2. 텍스트 내용 및 스타일 설정
                conf = float(best_box.conf)

                label = f"Tower: {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.9
                thickness = 2

                # 3. 텍스트 중앙 정렬 좌표 계산
                # 텍스트의 크기(너비, 높이)를 미리 계산
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # 박스의 너비 구하기
                box_w = x2 - x1

                # [X좌표] 박스 중앙 - 텍스트 너비 절반 (가로 중앙 정렬)
                text_x = int(x1 + (box_w - text_w) / 2)

                # [Y좌표] 박스 상단(y1) + 텍스트 높이 + 약간의 여백 (박스 안쪽 상단에 배치)
                text_y = int(y1 + text_h + 10)

                # 텍스트가 박스 왼쪽을 뚫고 나가면 보정
                if text_x < x1:
                    text_x = int(x1)

                # 4. 텍스트 그리기
                cv2.putText(img, label, (text_x, text_y),
                            font, font_scale, (0, 0, 255), thickness)

                # 5. 이미지 띄우기
                cv2.imshow("Debug Result", img)
                cv2.waitKey(0)  # 키 입력 대기
                cv2.destroyAllWindows()
        else:
            print("none")

    return 0


if __name__ == "__main__":
    sys.exit(main())