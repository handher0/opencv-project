import os
import re
import argparse
import cv2
import sys
import numpy as np

COIN_TEMPLATES = {
    10: [],
    50: [],
    100: [],
    500: []
}

orb = cv2.ORB_create(nfeatures=2000)

def parse_args():
    p = argparse.ArgumentParser("CV assignment runner")

    # Sample argument usage
    p.add_argument("--input", required=True, type=str, help="path to input image")

    return p.parse_args()


def resize_with_aspect_ratio(image, width):
    """
    이미지 너비를 지정값으로 고정하고 비율을 유지하며 리사이징
    지정값보다 작은 이미지는 원본을 반환
    """
    (h, w) = image.shape[:2]

    # 너비가 1000보다 작거나 같으면 원본 반환
    if w <= width:
        return image

    # 비율 계산
    r = width / float(w)
    dim = (width, int(h * r))

    # 리사이징
    #print(f"resizing : {w}x{h} -> {dim[0]}x{dim[1]}")
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def filter_nested_circles(circles):
    """
    자신보다 큰 원 안에 원점이 위치하는 원 모두 제거
    """
    if circles is None or len(circles) == 0:
        return []

    circles = circles[0, :]
    sorted_circles = sorted(circles, key=lambda c: c[2], reverse=True)

    final_circles = []
    for current_circle in sorted_circles:
        x1, y1, r1 = current_circle
        is_nested = False

        for other_circle in final_circles:
            x2, y2, r2 = other_circle

            # 두 원의 중심 간의 거리
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            # 1. current_circle(r1)이 other_circle(r2)보다 작아야 하고,
            # 2. current_circle의 중심이 other_circle의 반경 안에 있어야 함 (dist < r2)
            # 만족하면, 작은 원은 큰 원 내의 노이즈로 간주
            if r1 < r2 and dist < r2:
                is_nested = True
                break

        if not is_nested:
            final_circles.append(current_circle)

    return np.uint16(np.around(final_circles))

def load_templates(template_folder="templete", resize_dim=(100, 100)):
    """
    'templete' 폴더에서 템플릿 이미지를 로드 및
    리사이징한 후, (kp, des) 쌍을 COIN_TEMPLATES에 저장
    """
    if not os.path.exists(template_folder):
        #print(f"Error: Template folder not found at '{template_folder}'")
        return

    pattern = re.compile(r"(\d+)([fb])\.(jpe?g|png)$", re.IGNORECASE)
    #print(f"Loading templates (Resizing to {resize_dim})...")

    for filename in os.listdir(template_folder):
        match = pattern.match(filename)
        if not match: continue

        coin_value_str, side, extension = match.groups()
        coin_value = int(coin_value_str)
        if coin_value not in COIN_TEMPLATES: continue

        filepath = os.path.join(template_folder, filename)
        template_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if template_img is None:
            #print(f"Failed to load template: {filename}")
            continue

        try:
            template_resized = cv2.resize(template_img, resize_dim, interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            #print(f"Failed to resize template {filename}: {e}")
            continue

        kp, des = orb.detectAndCompute(template_resized, None)

        if des is not None and len(kp) > 0:
            # des만 저장하는 대신 (kp, des) 튜플(쌍)을 저장
            COIN_TEMPLATES[coin_value].append((kp, des))




def is_ten(color_roi_cropped, x, y, r, ratio_threshold=0.6):
    """
    입력된 컬러 ROI가 10원짜리 동전(구리색)인지 색상으로 판별

    :param color_roi_cropped: bitwise_and로 추출된 원형의 BGR 컬러 ROI
    :param ratio_threshold: 10원으로 판단하기 위한 픽셀 비율 임계값
    :return: 10원이면 True, 아니면 False
    """
    # 1. ROI가 비어있는지 확인
    if color_roi_cropped is None or color_roi_cropped.size == 0:
        return False

    # 2. ROI를 BGR에서 HSV 컬러 스페이스로 변환
    hsv = cv2.cvtColor(color_roi_cropped, cv2.COLOR_BGR2HSV)

    # 3. 10원짜리 동전의 구리색/갈색 범위 정의 (HSV)
    # hsv_tuner.py 사용하여 튜닝값 설정
    lower_copper = np.array([0, 90, 50])
    upper_copper = np.array([24, 255, 255])

    # 4. 지정된 색상 범위에 해당하는 픽셀 마스크 생성
    color_mask = cv2.inRange(hsv, lower_copper, upper_copper)

    # 5. ROI 영역(검은색 배경 제외)의 총 픽셀 수 계산
    gray_roi = cv2.cvtColor(color_roi_cropped, cv2.COLOR_BGR2GRAY)
    _ , coin_area_mask = cv2.threshold(gray_roi, 1, 255, cv2.THRESH_BINARY)
    total_coin_pixels = cv2.countNonZero(coin_area_mask)

    if total_coin_pixels == 0:
        return False  # 동전 영역이 없음

    # 6. 동전 영역 내에서 구리색 픽셀 수 계산
    copper_on_coin_mask = cv2.bitwise_and(color_mask, coin_area_mask)
    copper_pixel_count = cv2.countNonZero(copper_on_coin_mask)

    # 7. (디버깅용) 마스크 확인
    # cv2.imshow(f"Check 10w - {np.random.randint(0, 100)}",
    #            np.hstack([color_roi_cropped,
    #                       cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR),
    #                       cv2.cvtColor(copper_on_coin_mask, cv2.COLOR_GRAY2BGR)]))

    # 8. 동전 영역 대비 구리색 픽셀의 비율 계산
    copper_ratio = copper_pixel_count / float(total_coin_pixels)

    # 9. 비율이 임계값보다 높으면 10원으로 판단
    if copper_ratio > ratio_threshold:
        #print(f"  [10원 판정] (x:{x}, y:{y}, r:{r} / 비율: {copper_ratio:.2f})")
        return True
    else:
        # (디버깅용) 10원이 아니라고 판단될 때 비율 출력
        # print(f"  [10원 아님] (비율: {copper_ratio:.2f})")
        return False


# BFMatcher(Brute-Force Matcher) 생성
# NORM_HAMMING: ORB, BRIEF 같은 바이너리 디스크립터에 사용
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

def classify_coin(gray_roi_cropped, x, y, r,
                  resize_dim=(100, 100),
                  ratio_threshold=0.85,
                  min_inlier_threshold=4):
    """
    (RANSAC 적용)
    """
    #print(f"  [분류 시도] (x:{x}, y:{y}, r:{r})")

    # 1. 전처리: 리사이징 및 특징점 추출
    try:
        roi_normalized = cv2.resize(gray_roi_cropped, resize_dim, interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        #print(f"    -> [판정: 실패] (리사이징 오류: {e})")
        return 0

    # '입력 동전(ROI)'의 좌표와 지문
    kp_roi, des_roi = orb.detectAndCompute(roi_normalized, None)

    if des_roi is None or len(des_roi) < 2:
        #print(f"    -> [판정: 실패] (ROI에서 특징점 검출 실패)")
        return 0

    # 2. 점수표 생성 및 매칭
    scoreboard = {10: 0, 50: 0, 100: 0, 500: 0}
    min_good_matches_for_homography = 4

    for coin_value, template_list in COIN_TEMPLATES.items():

        current_coin_best_score = 0

        for kp_template, des_template in template_list:

            try:
                matches = bf.knnMatch(des_template, des_roi, k=2)
            except cv2.error:
                continue

            good_matches = []
            for match_pair in matches:
                if len(match_pair) < 2:
                    continue
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

            # 3. 기하학적 검증 (RANSAC)
            if len(good_matches) < min_good_matches_for_homography:
                continue

            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_roi[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            # --------------------

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

            if mask is None:
                continue

            inlier_count = np.sum(mask)

            if inlier_count > current_coin_best_score:
                current_coin_best_score = inlier_count

        scoreboard[coin_value] = current_coin_best_score
        #print(f"    - {coin_value}원 점수: {scoreboard[coin_value]} (Inliers)")

    # 4. 최종 판정
    best_match_value = max(scoreboard, key=scoreboard.get)
    best_match_score = scoreboard[best_match_value]

    if best_match_score >= min_inlier_threshold:
        #print(f"    -> [판정: {best_match_value}원] (Score: {best_match_score})")
        return best_match_value
    else:
        #print(f"    -> [판정: 실패] (최고 Inlier: {best_match_score} < {min_inlier_threshold})")
        return 0

def main():
    args = parse_args()

    total_amount = 0
    coin_counts = {500: 0, 100: 0, 50: 0, 10: 0}

    # 동전 검출
    # 1. 이미지 로드
    img = cv2.imread(args.input)
    if img is None:
        #print(f"Error: 이미지 로드 실패 {args.input}")
        return

    # 2. 리사이징
    img_resized = resize_with_aspect_ratio(img, width=800)
    output_img = img_resized.copy()

    # 3. 그레이스케일 변환
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    clahe_img = clahe.apply(gray)

    # 5. 가우시안 블러링
    blurred = cv2.GaussianBlur(clahe_img, (13, 13), 0)

    # 6. 캐니 엣지 출력 (디버깅용)
    canny_debug = cv2.Canny(blurred, 60, 120)

    # 7. 허프 변환 원 검출
    # dp: 누적기 해상도 비율 (1 = 원본 크기, 1.2 or 1.5 = 해상도 줄여 속도 향상)
    # minDist: 검출된 원 중심 간의 최소 거리 (동전 반지름보다 약간 크게)
    # param1: Canny 엣지 검출기의 상위 임계값
    # param2: 누적기 임계값 (작을수록 많은 원, 클수록 정확한 원 검출)
    # minRadius: 검출할 원의 최소 반지름
    # maxRadius: 검출할 원의 최대 반지름
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=50,
        param1=120,
        param2=45,
        minRadius=20,
        maxRadius=150
    )

    # 8. 원 안에 검출되는 원(노이즈) 제거
    if circles is not None:

        debug_output_img = img_resized.copy()
        raw_circles = np.uint16(np.around(circles))
        for i in raw_circles[0, :]:
            cv2.circle(debug_output_img, (i[0], i[1]), i[2], (0, 255, 255), 2)
            cv2.circle(debug_output_img, (i[0], i[1]), 2, (0, 0, 255), 3)
        count_text_raw = f"detection : {len(raw_circles[0])}"
        cv2.putText(debug_output_img, count_text_raw, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)

        final_circles = filter_nested_circles(circles)
        #print(f"초기 검출 원 개수: {len(circles[0])}, 최종 검출 원 개수: {len(final_circles)}")

        total_amount = 0

        # 동전 분류
        # 동전의 실제 물리적 크기 (지름 mm)
        COIN_DIAMETERS_MM = {
            500: 26.5,
            100: 24.0,
            50: 21.6,
            10: 18.0
        }

        if len(final_circles) > 0:

            # 1. 10원짜리 동전(Anchor) 찾기
            ten_won_circles = []
            other_circles = []

            for (x_u, y_u, r_u) in final_circles:
                x, y, r = int(x_u), int(y_u), int(r_u)

                # 원형 ROI 추출 (
                (h, w) = img_resized.shape[:2]
                mask = np.zeros((h, w), dtype="uint8")
                cv2.circle(mask, (x, y), r, 255, -1)

                # 'output_img' (컬러)에서 컬러 ROI 추출
                color_roi = cv2.bitwise_and(output_img, output_img, mask=mask)
                x1, y1 = max(0, x - r), max(0, y - r)
                x2, y2 = min(w, x + r), min(h, y + r)
                color_roi_cropped = color_roi[y1:y2, x1:x2]

                # HSV로 10원 판별
                if is_ten(color_roi_cropped, x, y, r):
                    ten_won_circles.append((x, y, r))
                else:
                    other_circles.append((x, y, r))

            # 2. 기준(px_per_mm) 설정
            px_per_mm = 0

            if len(ten_won_circles) > 0:
                # Case 1: 10원이 1개라도 발견됨
                # 10원 동전들의 평균 반지름 계산
                avg_r_10 = np.mean([r for (x, y, r) in ten_won_circles])
                # 10원(18.0mm) 기준으로 px_per_mm 확정
                px_per_mm = avg_r_10 / (COIN_DIAMETERS_MM[10] / 2.0)
                #print(f"--- [Anchor: 10원] ---")
                #print(f"Px/mm: {px_per_mm:.2f} (Based on 10won avg_r={avg_r_10:.1f})")

            else:
                #Case 2: 10원이 없음
                # 은색 동전 중 가장 큰 원을 500원으로 가정
                if len(other_circles) > 0:
                    max_r_silver = max([r for (x, y, r) in other_circles])
                    # 50원(21.6mm) 기준으로 px_per_mm 설정
                    px_per_mm = max_r_silver / (COIN_DIAMETERS_MM[500] / 2.0)
                    #print(f"--- [Anchor: 500원 (Fallback)] ---")
                    #print(f"Px/mm: {px_per_mm:.2f} (Based on 500won min_r={max_r_silver:.1f})")
                else:
                    #print("Error: No coins detected.")
                    return  # 분류 불가

            # 3. 확정된 px_per_mm로 모든 동전 분류

            # 예상 반지름 맵 계산
            expected_radii = {}
            for value, diameter in COIN_DIAMETERS_MM.items():
                expected_radii[value] = (diameter / 2.0) * px_per_mm

            # (디버깅) 예상 픽셀 크기 출력
            #print(f"Expected 500: {expected_radii[500]:.1f}px")
            #print(f"Expected 100: {expected_radii[100]:.1f}px")
            #print(f"Expected  50: {expected_radii[50]:.1f}px")
            #print(f"Expected  10: {expected_radii[10]:.1f}px")
            #print("----------------------------------")

            # 4. 모든 원을 순회하며 가장 가까운 크기의 동전으로 판정
            for (x_u, y_u, r_u) in final_circles:
                x, y, r = int(x_u), int(y_u), int(r_u)

                coin_value = 0
                min_error = float('inf')
                best_match = 0

                # 4-1. 가장 오차가 적은 동전(best_match)을 먼저 찾음
                for value, expected_r in expected_radii.items():
                    error = abs(r - expected_r)
                    if error < min_error:
                        min_error = error
                        best_match = value

                # 4-2. (튜닝포인트) 오차 임계값 (정밀하게 5%~7%로 설정)
                error_threshold_percent = 0.1  # 7% (5%는 너무 엄격할 수 있음)
                allowed_error_px = expected_radii[best_match] * error_threshold_percent

                # 4-3. 정밀 검사 및 경계 클리핑
                if min_error <= allowed_error_px:
                    # Case 1: 정밀 검사 통과
                    coin_value = best_match
                else:
                    # Case 2: 정밀 검사 실패
                    if r < expected_radii[10]:
                        coin_value = 10
                        #print(f"  (x:{x}, r:{r}) -> [Clipping] Too small, assigned 10")

                    elif r > expected_radii[500]:
                        coin_value = 500
                        #print(f"  (x:{x}, r:{r}) -> [Clipping] Too large, assigned 500")

                    else:
                        coin_value = 0

                total_amount += coin_value

                if coin_value in coin_counts:
                    coin_counts[coin_value] += 1

                    #coin_value에 따라 색상 동적 할당
                    color = (0, 255, 0)  # 초록
                    if coin_value == 10:
                        color = (0, 165, 255)  # 주황
                    elif coin_value == 50:
                        color = (255, 0, 0)  # 파랑
                    elif coin_value == 100:
                        color = (0, 0, 255)  # 빨강
                    elif coin_value == 500:
                        color = (255, 0, 255)  # 보라

                cv2.circle(output_img, (x, y), r, color, 3)
                text = str(coin_value) if coin_value > 0 else "?"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.putText(output_img, text, (x - text_w // 2, y + text_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            # 5. 검출된 동전 개수 및 총액 표시
            count_text = f"Detection: {len(final_circles)} coins"
            total_text = f"Total: {total_amount} KRW"

            cv2.putText(output_img, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output_img, total_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 0), 2, cv2.LINE_AA)

            # 전처리 과정 출력 (디버깅용)
            #cv2.imshow("original", img_resized)
            #cv2.imshow("CLAHE", clahe_img)
            #cv2.imshow("Blurred", blurred)
            #cv2.imshow("Debug Canny Edges", canny_debug)
            #cv2.imshow("Detection", debug_output_img)
            cv2.imshow("Result", output_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"500:{coin_counts[500]}")
        print(f"100:{coin_counts[100]}")
        print(f"50:{coin_counts[50]}")
        print(f"10:{coin_counts[10]}")
        print(total_amount)


if __name__ == "__main__":
    sys.exit(main())
