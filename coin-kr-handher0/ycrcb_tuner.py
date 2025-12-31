import cv2
import numpy as np
import argparse


def nothing(x):
    # 트랙바 콜백 함수 (아무것도 안 함)
    pass


def main():
    p = argparse.ArgumentParser("YCrCb Color Tuner")
    # (이전에 수정한 대로 --input을 사용합니다)
    p.add_argument("--input", required=True, type=str,
                   help="path to your test image (e.g., 10w_and_100w.jpg)")
    args = p.parse_args()

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

    # --- [수정] 이미지를 BGR에서 YCrCb로 변환 ---
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    cv2.namedWindow('Tuner')
    cv2.resizeWindow('Tuner', 500, 300)

    # --- [수정] 트랙바를 Y, Cr, Cb용으로 생성 (모두 0-255 범위) ---
    # Y = 밝기 (Luma), Cr = Red, Cb = Blue
    cv2.createTrackbar('Y_MIN', 'Tuner', 0, 255, nothing)
    cv2.createTrackbar('Cr_MIN', 'Tuner', 0, 255, nothing)
    cv2.createTrackbar('Cb_MIN', 'Tuner', 0, 255, nothing)
    cv2.createTrackbar('Y_MAX', 'Tuner', 255, 255, nothing)
    cv2.createTrackbar('Cr_MAX', 'Tuner', 255, 255, nothing)
    cv2.createTrackbar('Cb_MAX', 'Tuner', 255, 255, nothing)

    print("--- YCrCb Tuner ---")
    print("목표: 10원(구리색)은 '흰색', 100원(은색)은 '검은색'이 되도록 조절하세요.")
    print("팁: 'Y' (밝기)는 조명에 따라 변하므로 Y_MIN=0, Y_MAX=255로 두고,")
    print("    'Cr' (빨강)과 'Cb' (파랑) 값을 조절해 색을 찾으세요.")
    print("조절이 끝나면 'q' 키를 누르세요.")

    while True:
        # 트랙바에서 현재 값 가져오기
        y_min = cv2.getTrackbarPos('Y_MIN', 'Tuner')
        cr_min = cv2.getTrackbarPos('Cr_MIN', 'Tuner')
        cb_min = cv2.getTrackbarPos('Cb_MIN', 'Tuner')
        y_max = cv2.getTrackbarPos('Y_MAX', 'Tuner')
        cr_max = cv2.getTrackbarPos('Cr_MAX', 'Tuner')
        cb_max = cv2.getTrackbarPos('Cb_MAX', 'Tuner')

        lower = np.array([y_min, cr_min, cb_min])
        upper = np.array([y_max, cr_max, cb_max])

        # --- [수정] YCrCb 이미지에서 마스크 생성 ---
        mask = cv2.inRange(ycrcb, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow('Original Image', img)
        cv2.imshow('Mask (Result)', mask)  # <-- 가장 중요한 창
        cv2.imshow('Tuner Result', result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n--- Final YCrCb Values ---")
            print(f"lower_copper = np.array([{y_min}, {cr_min}, {cb_min}])")
            print(f"upper_copper = np.array([{y_max}, {cr_max}, {cb_max}])")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()