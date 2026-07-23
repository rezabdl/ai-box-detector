import cv2
import numpy as np

from app.image_processing import (
    detect_white_mask,
    detect_blue_mask,
    extract_contours,
)


class BoxDetector:
    """
    Detector untuk:
    1. Mencari kotak besar
    2. Mencari kotak-kotak kecil
    3. Mengklasifikasikan inside / outside

    Output akhir (method `detect`) HANYA berupa JSON summary, contoh:
    {
        "inside_box": 7,
        "outside_box": 2,
        "total": 9
    }
    """

    def __init__(self):

        # ==========================
        # Parameter Detector
        # ==========================

        self.MIN_BIG_AREA = 10000

        # -----------------------------------------------------
        # MIN_SMALL_AREA sedikit diturunkan (dari 1400 -> 1300).
        #
        # Kotak kecil yang posisinya bersinggungan dengan border
        # kotak besar kadang sedikit "terpotong" oleh garis hitam
        # tersebut, sehingga area yang terdeteksi sedikit lebih
        # kecil dari kotak-kotak lain (misal 1329 / 1392, padahal
        # kotak normal sekitar 1400-1500). Kotak-kotak ini BUKAN
        # noise ataupun pecahan dari kotak lain -- ukurannya
        # (lebar/tinggi) tetap mirip kotak normal, cuma areanya
        # sedikit di bawah ambang lama. Buffer kecil ini memberi
        # toleransi untuk kasus tersebut tanpa membuka pintu untuk
        # noise asli (yang biasanya areanya jauh lebih kecil, di
        # bawah 500).
        # -----------------------------------------------------
        self.MIN_SMALL_AREA = 1300
        self.MAX_SMALL_AREA = 3000

        self.MIN_RATIO = 0.50
        self.MAX_RATIO = 1.80

        self.MIN_EXTENT = 0.55

    # =====================================================
    # BIG RECTANGLE
    # =====================================================

    def find_big_rectangle(self, contours):
        """
        Mencari persegi panjang terbesar dari white mask.
        Return berupa polygon (4 titik).
        """

        best_polygon = None
        best_score = 0

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < self.MIN_BIG_AREA:
                continue

            # -----------------------------------------------------
            # Gunakan minAreaRect, bukan approxPolyDP, untuk
            # membentuk polygon 4 titik.
            #
            # Border kotak besar (kertas) sering digambar dengan
            # gaya sketsa: sudutnya tidak benar-benar 90 derajat
            # tajam, kadang ada garis dobel/bergerigi kecil di
            # ujungnya. Ditambah lagi rotasi kertas membuat tepi
            # kontur bergerigi akibat aliasing piksel. Kedua hal
            # ini membuat cv2.approxPolyDP() sering menghasilkan
            # >4 titik walaupun bentuknya secara visual tetap
            # persegi panjang, sehingga kandidat yang sebenarnya
            # valid ikut tertolak oleh pengecekan "len(polygon)
            # != 4".
            #
            # minAreaRect mencari persegi panjang (boleh miring)
            # yang paling pas membungkus contour, sehingga jauh
            # lebih toleran terhadap sudut yang tidak sempurna
            # maupun rotasi -- persis seperti yang sudah lebih
            # dulu diterapkan pada small box.
            # -----------------------------------------------------

            rect = cv2.minAreaRect(contour)
            (rect_w, rect_h) = rect[1]

            if rect_w == 0 or rect_h == 0:
                continue

            box_points = cv2.boxPoints(rect)
            polygon = np.int32(box_points).reshape(-1, 1, 2)

            rect_area = rect_w * rect_h
            extent = area / float(rect_area)

            # Ratio dihitung dari sisi asli minAreaRect
            # (sisi_pendek / sisi_panjang, rotation-invariant),
            # bukan dari bounding box axis-aligned yang bisa
            # bergeser mendekati 1.0 saat kertas diputar.
            long_side = max(rect_w, rect_h)
            short_side = min(rect_w, rect_h)
            ratio = short_side / float(long_side)

            # rasio kertas
            if ratio < 0.60 or ratio > 0.90:
                continue

            score = area * extent

            if score > best_score:
                best_score = score
                best_polygon = polygon

        if best_polygon is None:
            raise ValueError("Big rectangle tidak ditemukan.")

        return best_polygon

    # =====================================================
    # SMALL BOXES
    # =====================================================

    def find_small_boxes(self, contours):
        """
        Mencari seluruh kandidat kotak kecil.
        """

        small_boxes = []
        box_id = 1

        for contour_index, contour in enumerate(contours):

            area = cv2.contourArea(contour)

            # Contour yang benar-benar noise (nyaris tanpa luas)
            if area < 50:
                continue

            # -----------------------------------------------------
            # Gunakan minAreaRect, bukan approxPolyDP.
            #
            # Arsiran biru tidak selalu memiliki sudut setajam
            # border hitam (bisa sedikit membulat / bergerigi
            # akibat hatching), sehingga approxPolyDP sering
            # menghasilkan >4 titik dan kandidat yang sebenarnya
            # valid ikut terbuang. minAreaRect mencari persegi
            # (boleh miring) yang paling pas membungkus contour,
            # sehingga jauh lebih toleran terhadap ketidaksempurnaan
            # bentuk tapi tetap bisa memfilter blob yang bukan kotak.
            # -----------------------------------------------------

            rect = cv2.minAreaRect(contour)
            (rect_w, rect_h) = rect[1]

            if rect_w == 0 or rect_h == 0:
                continue

            box_points = cv2.boxPoints(rect)
            polygon = np.int32(box_points).reshape(-1, 1, 2)

            x, y, w, h = cv2.boundingRect(polygon)

            rect_area = rect_w * rect_h
            extent = area / float(rect_area)

            # Ratio disimpan sebagai sisi_panjang / sisi_pendek
            # (selalu >= 1) supaya tidak bergantung pada orientasi
            # rotasi kotak yang dikembalikan oleh minAreaRect.
            long_side = max(rect_w, rect_h)
            short_side = min(rect_w, rect_h)
            ratio = long_side / float(short_side)

            center = self.get_box_center(polygon)

            if area < self.MIN_SMALL_AREA:
                continue

            if area > self.MAX_SMALL_AREA:
                continue

            if ratio > self.MAX_RATIO:
                continue

            if extent < self.MIN_EXTENT:
                continue

            box = {
                "id": box_id,
                "contour_index": contour_index,
                "polygon": polygon,
                "area": area,
                "center": center,
                "bbox": (x, y, w, h),
                "ratio": ratio,
                "extent": extent
            }

            small_boxes.append(box)
            box_id += 1

        return small_boxes

    def merge_fragment_contours(
            self,
            contours,
            fragment_max_area=None,
            merge_distance=20
        ):
            """
            Menggabungkan pecahan kontur kecil (fragment) yang
            posisinya berdekatan menjadi satu kontur utuh.

            Hanya kontur dengan area kecil (di bawah
            fragment_max_area, default = MIN_SMALL_AREA) yang menjadi
            kandidat penggabungan. Kontur berukuran normal (kotak
            utuh) tidak pernah disentuh, sehingga kotak-kotak lain
            yang kebetulan berdekatan tetap aman terpisah.
            """

            if fragment_max_area is None:
                fragment_max_area = self.MIN_SMALL_AREA

            # Pisahkan dulu kontur "utuh" (sudah valid, tidak disentuh)
            # dari kontur "pecahan" (kandidat digabung).
            normal_contours = []
            fragments = []

            for contour in contours:

                area = cv2.contourArea(contour)

                if area < 50:
                    continue

                if area < fragment_max_area:
                    fragments.append(contour)
                else:
                    normal_contours.append(contour)

            merged_any = True

            while merged_any and len(fragments) > 1:

                merged_any = False

                for i in range(len(fragments)):
                    for j in range(i + 1, len(fragments)):

                        Mi = cv2.moments(fragments[i])
                        Mj = cv2.moments(fragments[j])

                        if Mi["m00"] == 0 or Mj["m00"] == 0:
                            continue

                        ci = (Mi["m10"] / Mi["m00"], Mi["m01"] / Mi["m00"])
                        cj = (Mj["m10"] / Mj["m00"], Mj["m01"] / Mj["m00"])

                        distance = (
                            (ci[0] - cj[0]) ** 2
                            + (ci[1] - cj[1]) ** 2
                        ) ** 0.5

                        if distance <= merge_distance:

                            combined_points = np.concatenate(
                                (fragments[i], fragments[j]), axis=0
                            )

                            merged_contour = cv2.convexHull(combined_points)

                            new_fragments = [
                                c for k, c in enumerate(fragments)
                                if k != i and k != j
                            ]
                            new_fragments.append(merged_contour)

                            fragments = new_fragments
                            merged_any = True
                            break

                    if merged_any:
                        break

            return normal_contours + fragments

    # =====================================================
    # SPLIT OVERSIZED CONTOURS (2 kotak menyatu)
    # =====================================================

    def split_merged_box(self, rect):
        """
        Membelah satu minAreaRect yang terlalu memanjang
        (indikasi dua kotak yang saling menempel) menjadi dua
        rectangle yang lebih kecil, dipotong tegak lurus
        terhadap sisi panjangnya.
        """

        (cx, cy), (rect_w, rect_h), angle = rect

        if rect_w >= rect_h:
            long_len = rect_w
            short_len = rect_h
            horizontal = True
        else:
            long_len = rect_h
            short_len = rect_w
            horizontal = False

        half_len = long_len / 2.0
        theta = np.deg2rad(angle)

        # Vektor arah sepanjang sisi panjang rectangle,
        # mengikuti sudut rotasi (angle) dari minAreaRect.
        if horizontal:
            dx, dy = np.cos(theta), np.sin(theta)
            size = (half_len, short_len)
        else:
            dx, dy = -np.sin(theta), np.cos(theta)
            size = (short_len, half_len)

        offset = half_len / 2.0

        center1 = (cx - dx * offset, cy - dy * offset)
        center2 = (cx + dx * offset, cy + dy * offset)

        rect1 = (center1, size, angle)
        rect2 = (center2, size, angle)

        return rect1, rect2

    def split_oversized_contours(self, contours):
        """
        Membelah kontur yang terindikasi dua kotak kecil yang
        saling menempel menjadi dua kontur terpisah.

        Indikasinya: rasio sisi panjang/pendek dari minAreaRect
        kontur tersebut melebihi MAX_RATIO, TAPI luasnya masih
        sekitar 1.5x - 2.2x luas kotak kecil normal (bukan noise,
        bukan juga satu kotak yang memang berbentuk memanjang).
        Ini pertanda kontur tersebut sebenarnya dua kotak yang
        bersentuhan dan ikut menyatu saat findContours, bukan
        benar-benar satu objek.
        """

        result = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 50:
                result.append(contour)
                continue

            rect = cv2.minAreaRect(contour)
            (rect_w, rect_h) = rect[1]

            if rect_w == 0 or rect_h == 0:
                result.append(contour)
                continue

            long_side = max(rect_w, rect_h)
            short_side = min(rect_w, rect_h)
            ratio = long_side / float(short_side)

            is_double_area = (
                self.MIN_SMALL_AREA * 1.5
                <= area
                <= self.MAX_SMALL_AREA * 2.2
            )

            if ratio > self.MAX_RATIO and is_double_area:

                rect1, rect2 = self.split_merged_box(rect)

                for sub_rect in (rect1, rect2):
                    box_points = cv2.boxPoints(sub_rect)
                    sub_polygon = np.int32(box_points).reshape(-1, 1, 2)
                    result.append(sub_polygon)

            else:
                result.append(contour)

        return result

    def filter_small_boxes(self, small_boxes):
        """
        Memfilter kandidat small box berdasarkan
        area, ratio, dan extent.
        """

        filtered_boxes = []

        for box in small_boxes:

            area = box["area"]
            ratio = box["ratio"]
            extent = box["extent"]

            if area < self.MIN_SMALL_AREA:
                continue

            if area > self.MAX_SMALL_AREA:
                continue

            if ratio > self.MAX_RATIO:
                continue

            if extent < self.MIN_EXTENT:
                continue

            filtered_boxes.append(box)

        return filtered_boxes

    def remove_duplicate_boxes(self, small_boxes, iou_threshold=0.80):
        """
        Menghapus box yang saling overlap menggunakan IoU.
        """

        unique_boxes = []

        for candidate in small_boxes:

            duplicate = False

            for saved in unique_boxes:

                iou = self.calculate_iou(candidate["bbox"], saved["bbox"])

                if iou >= iou_threshold:

                    duplicate = True

                    # Simpan box dengan area lebih besar
                    if candidate["area"] > saved["area"]:
                        unique_boxes.remove(saved)
                        unique_boxes.append(candidate)

                    break

            if not duplicate:
                unique_boxes.append(candidate)

        return unique_boxes

    # =====================================================
    # BOX INFORMATION
    # =====================================================

    def get_box_center(self, polygon):
        """
        Menghitung titik tengah polygon.
        """

        M = cv2.moments(polygon)

        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy)

    def calculate_iou(self, box1, box2):
        """
        Menghitung Intersection over Union (IoU)
        antara dua bounding box.
        """

        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)

        if right <= left or bottom <= top:
            return 0.0

        intersection = (right - left) * (bottom - top)

        area1 = w1 * h1
        area2 = w2 * h2

        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    # =====================================================
    # INSIDE TEST
    # =====================================================

    def is_box_inside(self, big_rectangle, polygon):
        """
        Mengecek apakah seluruh titik polygon
        berada di dalam big rectangle.
        """

        for point in polygon:

            x, y = point[0]

            result = cv2.pointPolygonTest(
                big_rectangle, (float(x), float(y)), False
            )

            if result < 0:
                return False

        return True

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    def classify_boxes(self, big_rectangle, small_boxes):
        """
        Mengelompokkan small box menjadi:
        - inside
        - outside
        """

        inside_boxes = []
        outside_boxes = []

        for box in small_boxes:

            inside = self.is_box_inside(big_rectangle, box["polygon"])

            if inside:
                inside_boxes.append(box)
            else:
                outside_boxes.append(box)

        return inside_boxes, outside_boxes

    # =====================================================
    # SUMMARY (JSON)
    # =====================================================

    def build_summary(self, inside_boxes, outside_boxes):
        """
        Membuat ringkasan hasil deteksi dalam bentuk dict
        sederhana, siap diubah ke JSON.

        Contoh hasil:
        {
            "inside_box": 7,
            "outside_box": 2,
            "total": 9
        }
        """

        inside_count = len(inside_boxes)
        outside_count = len(outside_boxes)

        return {
            "inside_box": inside_count,
            "outside_box": outside_count,
            "total": inside_count + outside_count
        }

    # =====================================================
    # MAIN DETECTOR
    # =====================================================

    def detect(self, image):
        """
        Pipeline utama pendeteksian box.

        Return HANYA berupa JSON summary, contoh:
        {
            "inside_box": 7,
            "outside_box": 2,
            "total": 9
        }
        """

        # 1. Big rectangle dari white mask
        white_mask = detect_white_mask(image)
        white_contours = extract_contours(white_mask, external_only=True)
        big_rectangle = self.find_big_rectangle(white_contours)

        # 2. Small boxes dari blue mask
        blue_mask = detect_blue_mask(image)
        blue_contours = extract_contours(blue_mask, external_only=True)

        blue_contours = self.merge_fragment_contours(blue_contours)
        blue_contours = self.split_oversized_contours(blue_contours)

        small_boxes = self.find_small_boxes(blue_contours)

        # 3. Filtering
        small_boxes = self.filter_small_boxes(small_boxes)
        small_boxes = self.remove_duplicate_boxes(small_boxes)

        # 4. Klasifikasi inside / outside
        inside_boxes, outside_boxes = self.classify_boxes(
            big_rectangle, small_boxes
        )

        # 5. Summary (JSON)
        summary = self.build_summary(inside_boxes, outside_boxes)

        return summary