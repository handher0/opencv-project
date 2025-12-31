import cv2
import numpy as np
import argparse


def nothing(x):
    # 트랙바 콜백 함수 (아무것도 안 함)
    pass


def main():
    p = argparse.ArgumentParser("HSV Color Tuner")
    p.add_argument("--input", required=True, type=str,
                   help="path to your test image (e.g., 10w_and_100w.jpg)")
    args = p.parse_args()

    # 이미지 로드
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Failed to load image {args.input}")
        return

    # 리사이징
    (h, w) = img.shape[:2]
    if w > 1000:
        r = 1000.0 / w
        dim = (1000, int(h * r))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # HSV로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 튜너 윈도우 생성
    cv2.namedWindow('Tuner')
    cv2.resizeWindow('Tuner', 500, 300)

    # 트랙바 생성 (H: 0-179, S/V: 0-255)
    cv2.createTrackbar('H_MIN', 'Tuner', 0, 179, nothing)
    cv2.createTrackbar('S_MIN', 'Tuner', 90, 255, nothing)
    cv2.createTrackbar('V_MIN', 'Tuner', 0, 255, nothing)
    cv2.createTrackbar('H_MAX', 'Tuner', 25, 179, nothing)
    cv2.createTrackbar('S_MAX', 'Tuner', 255, 255, nothing)
    cv2.createTrackbar('V_MAX', 'Tuner', 255, 255, nothing)

    print("--- HSV Tuner ---")
    print("Goal: 10원 동전은 '흰색', 100원 동전은 '검은색'이 되도록 6개 슬라이더를 조절하세요.")
    print("조절이 끝나면 'q' 키를 누르세요.")

    while True:
        # 트랙바에서 현재 값 가져오기
        h_min = cv2.getTrackbarPos('H_MIN', 'Tuner')
        s_min = cv2.getTrackbarPos('S_MIN', 'Tuner')
        v_min = cv2.getTrackbarPos('V_MIN', 'Tuner')
        h_max = cv2.getTrackbarPos('H_MAX', 'Tuner')
        s_max = cv2.getTrackbarPos('S_MAX', 'Tuner')
        v_max = cv2.getTrackbarPos('V_MAX', 'Tuner')

        # lower / upper 범위 생성
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # 마스크 생성
        mask = cv2.inRange(hsv, lower, upper)

        # 원본 이미지에 마스크 적용 (결과 확인용)
        result = cv2.bitwise_and(img, img, mask=mask)

        # 화면에 표시
        cv2.imshow('Original Image', img)
        cv2.imshow('Mask (Result)', mask)
        cv2.imshow('Tuner Result', result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n--- Final Values ---")
            print(f"lower_copper = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper_copper = np.array([{h_max}, {s_max}, {v_max}])")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()