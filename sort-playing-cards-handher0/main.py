import argparse
import cv2
import sys
import numpy as np
import os

# ==========================================
# 1. 설정 및 상수
# ==========================================
PATH_RANKS = "ranks"
PATH_SUITS = "suits"

CARD_WIDTH = 200
CARD_HEIGHT = 300

CARD_MIN_AREA = 2500  # 너무 작은 노이즈 제거
CARD_MAX_AREA = 500000  # 너무 큰 영역 제거

# ==========================================
# 2. 데이터 로더
# ==========================================
class CardData:
    def __init__(self, name, image):
        self.name = name
        self.image = image

def load_reference_images(path):
    """참조 이미지 로드"""
    images = []
    if not os.path.exists(path):
        return []

    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            name = filename.split('.')[0]  # 확장자 제거
            images.append(CardData(name, img))
    return images

# ==========================================
# 3. 검출 (Detection)
# ==========================================
def detect_cards_robust(image, debug=False):
    """
    여러 Block Size로 Adaptive Threshold를 시도하여
    가장 안정적인 컨투어 그룹을 찾아냅니다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 노이즈 제거를 위한 블러
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 시도할 파라미터 후보군 (홀수만 가능)
    candidate_params = [19, 17, 15, 13, 11, 9, 7, 5, 3]
    results = []

    img_h, img_w = image.shape[:2]
    img_area = img_w * img_h

    for bs in candidate_params:
        # 1. 적응형 이진화 (Block Size 가변)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, bs, 5)

        # 2. 모폴로지 (끊어진 선 잇기)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3. 컨투어 검출
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        areas = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # 면적 필터링 (너무 작거나 전체 화면 덮는 것 제외)
            if area > 2000 and area < (img_area * 0.95):
                peri = cv2.arcLength(cnt, True)
                # 꼭짓점 근사화
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                # 사각형(4점)이고 비율이 정상적인지 확인
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    # 카드는 세로로 길거나 가로로 길 수 있음 (0.5 ~ 2.0 허용)
                    if 0.5 < aspect_ratio < 2.0:
                        valid_contours.append(approx)
                        areas.append(area)

        count = len(valid_contours)
        if count > 0:
            # 점수 계산: (표준편차 / 평균) -> 크기가 균일할수록(변동계수가 낮을수록) 좋음
            mean_area = np.mean(areas)
            std_dev = np.std(areas)
            cv_val = std_dev / mean_area if mean_area > 0 else 0

            results.append({
                'bs': bs,
                'contours': valid_contours,
                'count': count,
                'cv': cv_val
            })

    # [최적의 결과 선택]
    # 1. 변동계수(cv)가 0.3 이하인(크기가 비슷한) 그룹만 필터링
    valid_results = [r for r in results if r['cv'] <= 0.3]

    if len(valid_results) > 0:
        # 2. 카드를 가장 많이 찾은 순서 -> 변동계수가 낮은 순서로 정렬
        best_result = sorted(valid_results, key=lambda x: (-x['count'], x['cv']))[0]
        if debug: print(f">> Best Param: BlockSize={best_result['bs']}, Count={best_result['count']}")
        return best_result['contours']
    elif len(results) > 0:
        # 조건 만족하는 게 없으면 그냥 제일 많이 찾은거라도 반환
        best_result = sorted(results, key=lambda x: -x['count'])[0]
        return best_result['contours']
    else:
        return []


def flatten_card(image, pts):
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [CARD_WIDTH - 1, 0],
        [CARD_WIDTH - 1, CARD_HEIGHT - 1],
        [0, CARD_HEIGHT - 1]], dtype="float32")

    temp_rect = np.array([tl, tr, br, bl], dtype="float32")
    M = cv2.getPerspectiveTransform(temp_rect, dst)

    warp_color = cv2.warpPerspective(image, M, (CARD_WIDTH, CARD_HEIGHT))

    if maxWidth > maxHeight:
        warp_color = cv2.rotate(warp_color, cv2.ROTATE_90_CLOCKWISE)
        warp_color = cv2.resize(warp_color, (CARD_WIDTH, CARD_HEIGHT))

    # ==========================================
    # ★ [수정 포인트] 마진을 2로 복구 (혹은 0)
    # ==========================================
    margin = 2  # 기존 5 -> 2로 변경 (글자 잘림 방지)

    if margin > 0:
        h, w = warp_color.shape[:2]
        crop_color = warp_color[margin:h - margin, margin:w - margin]
        warp_color = cv2.resize(crop_color, (CARD_WIDTH, CARD_HEIGHT))

    warp_gray = cv2.cvtColor(warp_color, cv2.COLOR_BGR2GRAY)

    return warp_gray, warp_color

# ==========================================
# 4. 인식 (Recognition) - Helpers
# ==========================================
def extract_roi_by_labeling(warp_image, debug=False, card_idx=0):
    H, W = warp_image.shape
    # 탐색 영역 제한 (35% / 25%)
    margin_x = 0
    search_h = int(H * 0.35)
    search_w = int(W * 0.25)
    crop = warp_image[0:search_h, margin_x:search_w]

    _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    valid_blobs = []
    for i in range(1, cnt):
        x, y, w, h, area = stats[i]
        if area < 30 or area > 4000: continue
        if x == 0 or (x + w) >= (search_w - margin_x): continue
        valid_blobs.append({'x': x + margin_x, 'y': y, 'w': w, 'h': h})

    if len(valid_blobs) < 1:
        return warp_image[5:60, 5:50], warp_image[60:110, 5:50]

    valid_blobs.sort(key=lambda b: b['y'])

    # Rank
    rank_blobs = [valid_blobs[0]]
    if len(valid_blobs) > 1:
        second = valid_blobs[1]
        y_diff = abs(valid_blobs[0]['y'] - second['y'])
        x_dist = second['x'] - (valid_blobs[0]['x'] + valid_blobs[0]['w'])
        if y_diff < 15 and x_dist < 15:
            rank_blobs.append(second)

    min_x = min([b['x'] for b in rank_blobs])
    min_y = min([b['y'] for b in rank_blobs])
    max_x = max([b['x'] + b['w'] for b in rank_blobs])
    max_y = max([b['y'] + b['h'] for b in rank_blobs])

    pad = 4
    r_y1 = max(0, min_y - pad);
    r_y2 = min(H, max_y + pad)
    r_x1 = max(0, min_x - pad);
    r_x2 = min(W, max_x + pad)
    img_rank_roi = warp_image[r_y1:r_y2, r_x1:r_x2]

    # --- 2. Suit ROI 추출 (레이블링 도입) ---
    suit_candidates = [b for b in valid_blobs if b['y'] > max_y]  # 랭크 아래에 있는 블롭만 후보

    s_y1 = -1
    s_y2 = -1
    s_x1 = -1
    s_x2 = -1

    if suit_candidates:
        # 수트 후보 선정 기준: (가장 상단에 있으면서, 랭크 X 중심에 가장 가까운 블롭)
        suit_candidates.sort(key=lambda b: b['y'])  # 다시 Y 좌표 순으로 정렬

        # 랭크 중심 X 좌표
        rank_center_x = (min_x + max_x) / 2

        # 1차 필터링: 랭크와 수평 위치가 너무 동떨어지지 않은 것 선택 (선택적)
        aligned_candidates = [b for b in suit_candidates if abs((b['x'] + b['w'] / 2) - rank_center_x) < 25]

        if aligned_candidates:
            best_suit = aligned_candidates[0]
        else:
            best_suit = suit_candidates[0]  # 정렬된 것 중 가장 위에 있는 것 선택 (기본 Fallback)

        s_y1 = best_suit['y']
        s_y2 = best_suit['y'] + best_suit['h']
        s_x1 = best_suit['x']
        s_x2 = best_suit['x'] + best_suit['w']

        # 최종 ROI 설정 (여백 추가)
        s_y1 = max(0, s_y1 - pad)
        s_y2 = min(H, s_y2 + pad)
        s_x1 = max(0, s_x1 - pad)
        s_x2 = min(W, s_x2 + pad)

        img_suit_roi = warp_image[s_y1:s_y2, s_x1:s_x2]
    else:
        # 후보가 없으면 기존의 정적 추정 방식을 최종 Fallback으로 사용
        # (이 코드는 원래 로직의 Fallback을 그대로 복사한 것)
        s_y1_fallback = max_y + 2
        s_y2_fallback = min(H, s_y1_fallback + 55)
        rank_center_x_fallback = (min_x + max_x) // 2
        s_x1_fallback = max(0, rank_center_x_fallback - 25)
        s_x2_fallback = min(W, rank_center_x_fallback + 25)
        img_suit_roi = warp_image[s_y1_fallback:s_y2_fallback, s_x1_fallback:s_x2_fallback]

    return img_rank_roi, img_suit_roi


def trim_image(img):
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None: return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y + h, x:x + w]


def resize_with_padding(image, target_size):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0: return cv2.resize(image, (max(1, target_w), max(1, target_h)))

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def get_augmented_templates(img, is_rank=False):
    if is_rank: return [img]
    templates = [img]
    kernel = np.ones((3, 3), np.uint8)
    templates.append(cv2.erode(img, kernel, iterations=1))
    templates.append(cv2.dilate(img, kernel, iterations=1))
    templates.append(cv2.dilate(img, kernel, iterations=2))
    templates.append(cv2.GaussianBlur(img, (3, 3), 0))
    return templates


def get_shape_features(roi):
    _, thresh = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0, 0.0
    main_cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_cnt)
    if area == 0: return 0, 0.0
    hull = cv2.convexHull(main_cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0.0
    holes = 0
    if hierarchy is not None:
        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1:
                if cv2.contourArea(contours[i]) > 10: holes += 1
    return holes, solidity


def refine_roi(roi, is_rank=True):
    if is_rank: return roi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)


def match_card(warp_gray, warp_color, train_ranks, train_suits, card_idx=0, debug=False):
    # 1. 색상 판별 (Red / Black / Ambiguous)
    card_color = get_card_color(warp_color)
    if debug: print(f"Card {card_idx} Color: {card_color}")

    # 2. ROI 추출 & 전처리
    img_rank_roi, img_suit_roi = extract_roi_by_labeling(warp_gray, debug, card_idx)
    _, img_rank_roi = cv2.threshold(img_rank_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, img_suit_roi = cv2.threshold(img_suit_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Suit 정제
    img_suit_roi = refine_roi(img_suit_roi, is_rank=False)

    # 기하학 특징 추출
    rank_holes, rank_solidity = get_shape_features(img_rank_roi)
    suit_holes, suit_solidity = get_shape_features(img_suit_roi)

    if debug:
        cv2.imshow(f"Rank ROI {card_idx}", img_rank_roi)
        cv2.imshow(f"Suit ROI {card_idx}", img_suit_roi)

    # =========================================================
    # ★ 10 (Ten) 즉시 확정 로직
    # =========================================================
    h, w = img_rank_roi.shape
    aspect_ratio = w / h if h > 0 else 0
    cnts, _ = cv2.findContours(img_rank_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob_count = sum(1 for c in cnts if cv2.contourArea(c) > 20)

    is_ten = (blob_count >= 2 and aspect_ratio > 0.70)

    if is_ten:
        r_name = "10"
        r_score = 1.0
        if debug: print(f"   [Instant] Rank -> 10")
    else:
        r_name = None

    # =========================================================
    # 내부 매칭 함수 (Ambiguous 처리 추가)
    # =========================================================
    def run_matching(roi, templates_list, is_suit=False):
        best_name = "Unknown"
        max_score = -1.0
        h, w = roi.shape

        for t_data in templates_list:
            t_name = t_data.name

            # ★ [수정됨] 색상 필터링 로직 완화
            if is_suit:
                # 색상이 'Ambiguous'면 필터링하지 않고 다 검사함 (Pass)
                # 확실한 경우에만 반대 색상 제외
                if card_color == "Red" and t_name in ['Spades', 'Clubs', 'S', 'C']:
                    continue
                if card_color == "Black" and t_name in ['Hearts', 'Diamonds', 'H', 'D']:
                    continue

            t_trimmed = trim_image(t_data.image)
            augmented_imgs = get_augmented_templates(t_trimmed, is_rank=not is_suit)

            temp_best_score = -1.0

            for t_img_aug in augmented_imgs:
                _, t_bin = cv2.threshold(t_img_aug, 127, 255, cv2.THRESH_BINARY)

                scale_factors = [1.0, 0.9]
                for scale in scale_factors:
                    if scale < 1.0:
                        th, tw = t_bin.shape
                        nw, nh = int(tw * scale), int(th * scale)
                        t_small = cv2.resize(t_bin, (nw, nh), interpolation=cv2.INTER_AREA)
                        t_final_base = t_small
                    else:
                        t_final_base = t_bin

                    t_final = resize_with_padding(t_final_base, (w, h))
                    res = cv2.matchTemplate(roi, t_final, cv2.TM_CCOEFF_NORMED)
                    score = res[0][0]
                    if score > temp_best_score: temp_best_score = score

            final_score = temp_best_score

            if 'Eight' in t_name or '8' == t_name:
                if rank_holes < 2: final_score -= 0.2
            elif t_name in ['Queen', 'Q', 'Nine', '9', 'Six', '6', 'Four', '4', 'Zero', '0', 'Ace', 'A']:
                if rank_holes != 1: final_score -= 0.1

            if final_score > max_score:
                max_score = final_score
                best_name = t_data.name

        return best_name, max_score

    # Rank 매칭
    if not is_ten:
        r_name, r_score = run_matching(img_rank_roi, train_ranks, is_suit=False)
        # 6, 9, Q 검증
        if r_name in ['6', '9', 'Q', 'Six', 'Nine', 'Queen'] and r_score > 0.2:
            corrected_name = verify_69Q(img_rank_roi, r_name)
            if corrected_name != r_name:
                if debug: print(f"   [Geo Fix] {r_name} -> {corrected_name}")
                r_name = corrected_name

    # Suit 매칭 (Ambiguous면 4개 다 비교함)
    s_name, s_score = run_matching(img_suit_roi, train_suits, is_suit=True)

    # Suit 기하학 검증
    if s_name != "Unknown" and s_score > 0.2:
        corrected_suit = verify_suit(img_suit_roi, s_name)
        if corrected_suit != s_name:
            if debug: print(f"   [Geo Fix] {s_name} -> {corrected_suit}")
            s_name = corrected_suit

    threshold_score = 0.2
    if not is_ten and r_score < threshold_score: r_name = "Unknown"
    if s_score < threshold_score: s_name = "Unknown"

    if debug:
        print(f"   -> Result: {r_name}({r_score if not is_ten else 1.0:.2f}) {s_name}({s_score:.2f})")

    return r_name, s_name


def get_card_color(img_color):
    """
    [수정됨] 색상 판별 로직 (3단계)
    - Red: 확실히 빨감
    - Black: 확실히 검음 (붉은기 거의 없음)
    - Ambiguous: 애매함 (조명이나 노이즈 때문일 수 있음) -> 4개 문양 다 검사해야 함
    """
    h, w = img_color.shape[:2]
    # 코너 영역 설정 (좌측 상단)
    corner = img_color[0:int(h * 0.4), 0:int(w * 0.3)]

    hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)

    # Red 범위 (넓게 잡음)
    lower_red1 = np.array([0, 40, 20]);
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 40, 20]);
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    red_pixels = cv2.countNonZero(mask)
    ratio = red_pixels / (corner.shape[0] * corner.shape[1])

    # [3단계 판별]
    if ratio > 0.7:  # 3% 이상이면 확실히 Red
        return "Red"
    elif ratio < 0.1:  # 1% 미만이면 확실히 Black
        return "Black"
    else:  # 1% ~ 3% 사이는 애매함
        return "Ambiguous"


def verify_suit(roi, current_name):
    """
    [Suit 정밀 검증 - 최종 개선판]
    스페이드 vs 클럽: '최상단(15%)의 뾰족함'을 비교 (스페이드는 뾰족, 클럽은 뭉툭)
    하트 vs 다이아: '어깨 너비'와 '파임'을 비교
    """
    # 1. 노이즈 제거 및 잉크 추출 (Tight Crop)
    _, thresh = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None: return current_name

    x, y, w, h = cv2.boundingRect(coords)
    roi_tight = thresh[y:y + h, x:x + w]
    h_t, w_t = roi_tight.shape

    # 각 행(Row)의 너비 계산 (Width Profile)
    row_counts = [cv2.countNonZero(row) for row in roi_tight]
    max_width = max(row_counts) if row_counts else 1

    # ---------------------------------------------------------
    # Case A: 스페이드(Spades) vs 클럽(Clubs) [로직 수정됨]
    # ---------------------------------------------------------
    if current_name in ['Spades', 'Clubs', 'S', 'C']:
        # 전략: "최상단(Top 15%)이 뾰족한가(좁은가)?"
        # 스페이드는 뾰족해서 좁고, 클럽은 동그란 잎이라서 넓음.

        target_y = int(h_t * 0.15)  # 상단 15% 지점
        top_width = row_counts[target_y] if target_y < h_t else 0

        # 전체 뚱뚱한 정도 대비 머리의 너비 비율
        ratio = top_width / max_width

        # [판단 기준]
        # 스페이드: 창끝이라 매우 좁음 (비율 < 0.35)
        # 클럽: 둥근 잎이라 꽤 넓음 (비율 > 0.35)

        if ratio > 0.4:
            return 'Clubs'
        else:
            return 'Spades'

    # ---------------------------------------------------------
    # Case B: 하트(Hearts) vs 다이아몬드(Diamonds) (★ 신규 로직)
    # ---------------------------------------------------------
    elif current_name in ['Hearts', 'Diamonds', 'H', 'D']:
        # 전략: "어깨 너비" vs "허리 너비" 비율 비교

        # 1. 허리 너비 (높이 50% 지점) - 둘 다 가장 넓은 곳
        mid_y = int(h_t * 0.5)
        mid_row = roi_tight[mid_y, :]
        width_mid = cv2.countNonZero(mid_row)

        # 2. 어깨 너비 (높이 25% 지점) - 하트는 넓고, 다이아는 좁음
        shoulder_y = int(h_t * 0.25)
        shoulder_row = roi_tight[shoulder_y, :]
        width_shoulder = cv2.countNonZero(shoulder_row)

        # 0으로 나누기 방지
        if width_mid == 0: width_mid = 1

        # 비율 계산
        shoulder_ratio = width_shoulder / width_mid

        # [판단 기준]
        # 다이아몬드: 마름모꼴이라 위로 갈수록 급격히 좁아짐 (비율 0.5 ~ 0.6)
        # 하트: 위쪽 잎사귀 부분이 빵빵함 (비율 0.8 ~ 1.0)

        # 기준값: 0.75 (이보다 크면 하트)
        if shoulder_ratio > 0.75:
            return 'Hearts'
        else:
            # 혹시 모르니 기존의 '중앙 파임' 체크도 보조로 사용 (더블 체크)
            # 다이아몬드로 판단했지만, 만약 가운데가 깊게 파여있다면 하트일 수도 있음
            check_y_start = int(h_t * 0.1)
            check_y_end = int(h_t * 0.25)
            center_x = w_t // 2
            margin = int(w_t * 0.1)
            center_region = roi_tight[check_y_start:check_y_end, center_x - margin:center_x + margin]
            fill_ratio = cv2.countNonZero(center_region) / (center_region.size + 1e-5)

            # 어깨는 좁지만(다이아 특징), 가운데가 텅 비었다(하트 특징)? -> 하트로 구제
            if fill_ratio < 0.3:
                return 'Hearts'

            return 'Diamonds'

    return current_name

def verify_69Q(roi, current_name):
    """
    [6, 9, Q 구분 로직 - 사용자 커스텀]
    규칙:
    1. 이 덱의 Q는 구멍이 2개다. -> 구멍 >= 2면 무조건 Q
    2. 구멍이 1개면 위치로 판단 -> 위쪽(9), 아래쪽(6)
    """
    # 1. 노이즈 제거 및 이진화
    _, thresh = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)

    # 2. 컨투어 및 계층구조 찾기 (구멍 찾기용)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return current_name  # 구멍 정보 없으면 원래 예측 유지

    # 3. 유효한 '구멍' 개수 세기
    holes = []
    roi_h, roi_w = roi.shape

    for i in range(len(contours)):
        # hierarchy: [Next, Prev, First_Child, Parent]
        parent_idx = hierarchy[0][i][3]

        # 부모가 있고(==구멍), 노이즈가 아닌(면적이 적당히 큰) 것
        if parent_idx != -1:
            area = cv2.contourArea(contours[i])
            if area > 1:  # 최소 면적 제한 (노이즈 방지)
                holes.append(contours[i])

    # --- 판정 로직 ---

    # Case 1: 구멍이 2개 이상이다 -> 무조건 Q
    if len(holes) >= 2:
        return 'Q'

    # Case 2: 구멍이 1개다 -> 6 vs 9 (위치로 승부)
    elif len(holes) == 1:
        hole_cnt = holes[0]

        # 구멍의 무게중심(Center Y) 계산
        M = cv2.moments(hole_cnt)
        if M['m00'] == 0: return current_name

        cy = int(M['m01'] / M['m00'])

        # 높이 대비 위치 비율 (0.0 ~ 1.0)
        ratio_y = cy / roi_h

        if ratio_y < 0.5:  # 구멍이 위쪽(절반보다 위) -> 9
            return '9'
        else:  # 구멍이 아래쪽(절반보다 아래) -> 6
            return '6'

    # Case 3: 구멍이 없다?
    # 잉크가 뭉쳐서 막혔거나 폰트 특성상 Q일 확률이 높음 (6, 9는 구멍이 큼)
    else:
        return 'Q'


# ==========================================
# 5. 유틸리티 및 메인
# ==========================================
def resize_image(image, target_width=None):
    if target_width is None: return image
    (h, w) = image.shape[:2]
    r = target_width / float(w)
    dim = (target_width, int(h * r))
    interpolation = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, dim, interpolation=interpolation)


def get_sort_key(card_str):
    suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
    rank_map = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12,
                'K': 13}
    suit_char = card_str[0]
    rank_str = card_str[1:]
    return (rank_map.get(rank_str, 0) * 10) + suit_map.get(suit_char, 0)


def convert_to_output_format(rank_name, suit_name):
    r_map = {'Ace': 'A', 'Two': '2', 'Three': '3', 'Four': '4', 'Five': '5', 'Six': '6', 'Seven': '7', 'Eight': '8',
             'Nine': '9', 'Ten': '10', 'Jack': 'J', 'Queen': 'Q', 'King': 'K', 'A': 'A', '2': '2', '3': '3', '4': '4',
             '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '10': '10', 'J': 'J', 'Q': 'Q', 'K': 'K'}
    s_map = {'Clubs': 'C', 'Diamonds': 'D', 'Hearts': 'H', 'Spades': 'S', 'C': 'C', 'D': 'D', 'H': 'H', 'S': 'S'}
    r = r_map.get(rank_name, '?')
    s = s_map.get(suit_name, '?')
    return s + r


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str)
    return p.parse_args()


def main():
    args = parse_args()
    DEBUG = False

    img = cv2.imread(args.input)
    if img is None: return -1

    # 리사이징
    img = resize_image(img, target_width=1000)

    train_ranks = load_reference_images(PATH_RANKS)
    train_suits = load_reference_images(PATH_SUITS)
    if not train_ranks or not train_suits: return 0

    # 1. 카드 검출
    best_contours = detect_cards_robust(img, debug=DEBUG)

    # [디버깅용 이미지]
    debug_img = img.copy()

    detected_cards = []

    for i, cnt in enumerate(best_contours):
        # ★ [수정됨] flatten_card가 2개를 반환하므로 변수 2개로 받음
        warp_gray, warp_color = flatten_card(img, cnt)

        # ★ [수정됨] imshow에는 이미지 하나만 넣어야 함 (warp -> warp_gray)
        if DEBUG: cv2.imshow(f"Debug Warped {i + 1}", warp_gray)

        # ★ [수정됨] match_card에 두 이미지 모두 전달
        r_name, s_name = match_card(warp_gray, warp_color, train_ranks, train_suits, card_idx=i + 1, debug=DEBUG)

        # 4. 결과 저장 및 시각화
        if r_name != "Unknown" and s_name != "Unknown":
            fmt_card = convert_to_output_format(r_name, s_name)
            if '?' not in fmt_card:
                detected_cards.append(fmt_card)

                # 시각화 (테두리 + 텍스트)
                cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.putText(debug_img, fmt_card, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                # 포맷 에러 시
                cv2.drawContours(debug_img, [cnt], -1, (0, 255, 255), 2)
        else:
            # 인식 실패 시
            cv2.drawContours(debug_img, [cnt], -1, (100, 100, 100), 2)

    # 5. 정렬 및 출력
    detected_cards.sort(key=get_sort_key)
    print(" ".join(detected_cards))

    # 최종 결과 창
    if DEBUG:
        cv2.namedWindow("Result Visualization", cv2.WINDOW_NORMAL)
        cv2.imshow("Result Visualization", debug_img)
        print(">> 모든 작업 완료. 결과 창을 확인하고 아무 키나 누르면 종료됩니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())


