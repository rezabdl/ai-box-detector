"""
app/image_processing.py
========================

Kumpulan fungsi utilitas untuk pemrosesan gambar yang dipakai oleh
`BoxDetector` (lihat app/detector.py):

- Load gambar
- Konversi warna (gray, HSV)
- Operasi morfologi (opening, closing, fill holes)
- Deteksi mask (putih, biru, hitam)
- Ekstraksi & menggambar contour
- Visualisasi gambar (khusus debugging manual, tidak dipakai
  di pipeline utama)

Semua fungsi di sini bersifat stateless (tidak menyimpan state),
sehingga bisa dipanggil langsung tanpa perlu membuat object apa pun.
"""

import cv2
import numpy as np


# ==========================================================
# Constants
# ==========================================================

# Ambang batas grayscale untuk mendeteksi area putih (kertas).
WHITE_THRESHOLD = 200

# Rentang HSV untuk warna biru (arsiran kotak kecil).
BLUE_LOWER = np.array([85, 40, 40])
BLUE_UPPER = np.array([125, 255, 255])

# Rentang HSV untuk warna hitam (border kotak besar).
BLACK_LOWER = np.array([0, 0, 0])
BLACK_UPPER = np.array([180, 255, 70])

# Kernel untuk Gaussian blur sebelum deteksi warna biru.
GAUSSIAN_KERNEL = (5, 5)

# Kernel morfologi dengan berbagai ukuran, dipakai sesuai
# kebutuhan masing-masing mask.
MORPH_KERNEL_SMALL = (3, 3)
MORPH_KERNEL_MEDIUM = (5, 5)
MORPH_KERNEL_LARGE = (7, 7)


# ==========================================================
# Image Loading
# ==========================================================

def load_image(image_path: str) -> np.ndarray:
    """
    Membaca gambar dari path menggunakan OpenCV.

    Parameters
    ----------
    image_path : str
        Path menuju file gambar (relatif atau absolut).

    Returns
    -------
    np.ndarray
        Gambar dalam format BGR (default OpenCV).

    Raises
    ------
    FileNotFoundError
        Jika gambar tidak ditemukan / gagal dibaca di path tersebut.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Gambar tidak ditemukan : {image_path}"
        )

    return image


# ==========================================================
# Color Conversion
# ==========================================================

def convert_to_gray(image: np.ndarray) -> np.ndarray:
    """
    Mengubah gambar BGR menjadi grayscale (1 channel).

    Parameters
    ----------
    image : np.ndarray
        Gambar input dalam format BGR.

    Returns
    -------
    np.ndarray
        Gambar grayscale.
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Mengubah gambar BGR menjadi HSV.

    HSV dipakai karena lebih stabil untuk segmentasi warna
    (biru, hitam) dibanding RGB/BGR, terutama terhadap variasi
    pencahayaan.

    Parameters
    ----------
    image : np.ndarray
        Gambar input dalam format BGR.

    Returns
    -------
    np.ndarray
        Gambar dalam format HSV.
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# ==========================================================
# Morphology
# ==========================================================

def apply_morphology(
    mask: np.ndarray,
    kernel_size=(5, 5),
    open_iter=1,
    close_iter=2,
    close_kernel_size=None
) -> np.ndarray:
    """
    Membersihkan noise pada binary mask menggunakan
    operasi Opening (erosi lalu dilasi) + Closing
    (dilasi lalu erosi).

    Opening : membuang noise kecil / titik-titik terpisah.
    Closing : menutup lubang kecil di dalam objek dan
              menyatukan bagian yang terputus-putus.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (hasil threshold / inRange) yang akan dibersihkan.
    kernel_size : tuple, default (5, 5)
        Ukuran kernel untuk opening (dan closing, jika
        close_kernel_size tidak diisi).
    open_iter : int, default 1
        Jumlah iterasi opening.
    close_iter : int, default 2
        Jumlah iterasi closing.
    close_kernel_size : tuple, optional
        Ukuran kernel khusus untuk closing, berbeda dari opening.
        Berguna saat closing perlu kernel lebih besar untuk
        menyatukan pola arsiran/hatching, tanpa membuat opening
        ikut menghapus garis tipis.

    Returns
    -------
    np.ndarray
        Mask yang sudah dibersihkan.
    """

    kernel = np.ones(kernel_size, np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=open_iter
    )

    close_kernel = (
        np.ones(close_kernel_size, np.uint8)
        if close_kernel_size is not None
        else kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=close_iter
    )

    return mask


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Mengisi lubang/rongga di dalam objek pada binary mask,
    menggunakan teknik flood fill dari luar objek lalu
    membalik hasilnya (flood fill inverse).

    Berguna misalnya saat area di dalam kotak arsiran biru
    ada bagian putih (rongga) yang perlu dianggap sebagai
    bagian dari kotak yang sama.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask 1 channel (nilai 0 atau 255).

    Returns
    -------
    np.ndarray
        Mask dengan rongga di dalam objek sudah terisi.
    """

    h, w = mask.shape

    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)

    return mask | flood_inv


# ==========================================================
# Mask Detection
# ==========================================================

def detect_white_mask(image: np.ndarray) -> np.ndarray:
    """
    Membuat binary mask untuk area putih (kertas).
    Dipakai untuk mencari Big Rectangle.

    Alur:
    1. Convert ke grayscale.
    2. Threshold biner dengan ambang WHITE_THRESHOLD.
    3. Morphology (opening + closing) kernel besar untuk
       membersihkan noise dan menyatukan area putih yang
       terputus-putus.

    Parameters
    ----------
    image : np.ndarray
        Gambar input dalam format BGR.

    Returns
    -------
    np.ndarray
        Binary mask area putih.
    """

    gray = convert_to_gray(image)

    _, mask = cv2.threshold(
        gray,
        WHITE_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    mask = apply_morphology(mask, kernel_size=MORPH_KERNEL_LARGE)

    return mask


def detect_black_mask(image: np.ndarray) -> np.ndarray:
    """
    Membuat binary mask untuk area hitam (border kotak besar).

    Catatan: mask ini saat ini hanya dipakai untuk kebutuhan
    debug/visualisasi manual, tidak dipakai dalam perhitungan
    inside/outside pada pipeline utama `BoxDetector.detect()`.

    Parameters
    ----------
    image : np.ndarray
        Gambar input dalam format BGR.

    Returns
    -------
    np.ndarray
        Binary mask area hitam.
    """

    hsv = convert_to_hsv(image)

    mask = cv2.inRange(hsv, BLACK_LOWER, BLACK_UPPER)

    mask = apply_morphology(mask, kernel_size=MORPH_KERNEL_SMALL)

    return mask


def detect_blue_mask(image: np.ndarray) -> np.ndarray:
    """
    Membuat binary mask untuk arsiran biru.
    Dipakai untuk mencari Small Boxes.

    Alur:
    1. Convert ke HSV lalu Gaussian blur (meredam noise warna
       sebelum thresholding).
    2. inRange dengan rentang BLUE_LOWER..BLUE_UPPER.
    3. Closing (tanpa opening) kernel 7x7 untuk menyatukan
       garis-garis hatch yang terputus-putus menjadi satu blob
       utuh per kotak.
    4. Fill holes untuk mengisi rongga di dalam blob tanpa
       memperbesar ukurannya, supaya kotak yang berdekatan
       tetap terpisah.

    Catatan penting: opening SENGAJA tidak dipakai di sini karena
    garis arsiran (hatching) cukup tipis -- opening (yang melakukan
    erosi lebih dulu) berisiko menghapus garis tersebut sebelum
    sempat disatukan oleh closing. Noise kecil yang tersisa akan
    otomatis terbuang lewat filter area di tahap deteksi kandidat
    box (lihat `BoxDetector.find_small_boxes`).

    Parameters
    ----------
    image : np.ndarray
        Gambar input dalam format BGR.

    Returns
    -------
    np.ndarray
        Binary mask arsiran biru.
    """

    hsv = convert_to_hsv(image)
    hsv = cv2.GaussianBlur(hsv, GAUSSIAN_KERNEL, 0)

    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    mask = apply_morphology(
        mask,
        kernel_size=(7, 7),
        open_iter=0,
        close_iter=1
    )

    mask = fill_holes(mask)

    return mask


# ==========================================================
# Contour Processing
# ==========================================================

def extract_contours(mask: np.ndarray, external_only=True):
    """
    Mengekstrak contour dari binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask sumber contour.
    external_only : bool, default True
        Jika True, hanya mengambil contour terluar (RETR_EXTERNAL).
        Jika False, mengambil seluruh hierarki contour (RETR_TREE).

    Returns
    -------
    list
        List of contours (masing-masing berupa np.ndarray titik).
    """

    mode = cv2.RETR_EXTERNAL if external_only else cv2.RETR_TREE

    contours, _ = cv2.findContours(
        mask,
        mode,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def draw_contours(
    image: np.ndarray,
    contours,
    color=(0, 255, 0),
    thickness=2
) -> np.ndarray:
    """
    Menggambar seluruh contour di atas salinan gambar
    (untuk keperluan debug/visualisasi).

    Parameters
    ----------
    image : np.ndarray
        Gambar dasar (tidak dimodifikasi langsung, akan di-copy).
    contours : list
        List of contours yang akan digambar.
    color : tuple, default (0, 255, 0)
        Warna garis dalam format BGR.
    thickness : int, default 2
        Ketebalan garis.

    Returns
    -------
    np.ndarray
        Salinan gambar dengan contour tergambar di atasnya.
    """

    output = image.copy()

    cv2.drawContours(output, contours, -1, color, thickness)

    return output


# ==========================================================
# Visualization (khusus debugging manual / lokal)
# ==========================================================

def show_image(title: str, image: np.ndarray) -> None:
    """
    Menampilkan satu gambar dalam window OpenCV.
    Hanya untuk debugging manual di lingkungan lokal
    (butuh display, tidak untuk dipanggil di pipeline/production).

    Parameters
    ----------
    title : str
        Judul window.
    image : np.ndarray
        Gambar yang akan ditampilkan.
    """

    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_multiple_images(images: dict) -> None:
    """
    Menampilkan beberapa gambar sekaligus, masing-masing
    dalam window terpisah. Hanya untuk debugging manual.

    Parameters
    ----------
    images : dict
        Dict {judul_window: gambar}.
    """

    for title, image in images.items():
        cv2.imshow(title, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()