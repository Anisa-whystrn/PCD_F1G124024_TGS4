import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import os
import glob

# ============================================================
# KONFIGURASI
# ============================================================

# Folder tempat 9 gambar
FOLDER_CITRA = "citra"

# Folder hasil
FOLDER_HASIL = "hasil"

os.makedirs(FOLDER_HASIL, exist_ok=True)




X1 = 0.04
Y1 = 0.87
X2 = 0.35
Y2 = 0.98


# ============================================================
# FUNGSI CROP NOMOR IJAZAH
# ============================================================

def crop_nomor_ijazah(img):

    tinggi, lebar = img.shape[:2]

    x1 = int(X1 * lebar)
    y1 = int(Y1 * tinggi)

    x2 = int(X2 * lebar)
    y2 = int(Y2 * tinggi)

    crop = img[y1:y2, x1:x2]

    return crop


# ============================================================
# 1. BRIGHTNESS ADJUSTMENT
# ============================================================

def brightness_adjustment(gray):

    brightness = 40

    hasil = cv2.convertScaleAbs(
        gray,
        alpha=1.0,
        beta=brightness
    )

    return hasil


# ============================================================
# 2. CONTRAST STRETCHING
# ============================================================

def contrast_stretching(gray):

    min_val = np.min(gray)
    max_val = np.max(gray)

    if max_val == min_val:
        return gray.copy()

    hasil = (
        (gray.astype(np.float32) - min_val)
        * 255
        / (max_val - min_val)
    )

    return hasil.astype(np.uint8)


# ============================================================
# 3. HISTOGRAM EQUALIZATION
# ============================================================

def histogram_equalization(gray):

    hasil = cv2.equalizeHist(gray)

    return hasil


# ============================================================
# OCR
# ============================================================

def lakukan_ocr(image):

    # Perbesar gambar
    image = cv2.resize(
        image,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    teks = pytesseract.image_to_string(
        image,
        lang="ind",
        config="--psm 7"
    )

    return teks.strip()


# ============================================================
# HISTOGRAM
# ============================================================

def buat_histogram(image):

    histogram = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    )

    return histogram


# ============================================================
# CARI SEMUA GAMBAR
# ============================================================

files = []

files += glob.glob(
    os.path.join(FOLDER_CITRA, "*.jpg")
)

files += glob.glob(
    os.path.join(FOLDER_CITRA, "*.jpeg")
)

files += glob.glob(
    os.path.join(FOLDER_CITRA, "*.png")
)

files += glob.glob(
    os.path.join(FOLDER_CITRA, "*.JPG")
)

files += glob.glob(
    os.path.join(FOLDER_CITRA, "*.PNG")
)

files.sort()


print("=" * 70)
print("PROGRAM ANALISIS NOMOR IJAZAH")
print("=" * 70)

print("Jumlah gambar:", len(files))

if len(files) == 0:

    print("Tidak ada gambar di folder citra!")

    exit()


# ============================================================
# PROSES SEMUA GAMBAR
# ============================================================

rekap = []


for nomor, file in enumerate(files, start=1):

    print("\n")
    print("=" * 70)
    print(f"GAMBAR {nomor}: {file}")
    print("=" * 70)


    # --------------------------------------------------------
    # BACA GAMBAR
    # --------------------------------------------------------

    img = cv2.imread(file)

    if img is None:

        print("Gagal membaca gambar!")

        continue


    # --------------------------------------------------------
    # PUTAR GAMBAR 90 DERAJAT
    # --------------------------------------------------------
    # Gambar contoh kamu posisi awalnya miring.
    #
    # ROTATE_90_CLOCKWISE = putar 90 derajat searah jarum jam

    img_rotated = cv2.rotate(
        img,
        cv2.ROTATE_90_CLOCKWISE
    )


    # --------------------------------------------------------
    # CROP NOMOR IJAZAH
    # --------------------------------------------------------

    roi = crop_nomor_ijazah(
        img_rotated
    )


    if roi.size == 0:

        print("Crop kosong!")

        continue


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # ENHANCEMENT
    # --------------------------------------------------------

    bright = brightness_adjustment(
        gray
    )

    contrast = contrast_stretching(
        gray
    )

    equalized = histogram_equalization(
        gray
    )


    # --------------------------------------------------------
    # NAMA FILE
    # --------------------------------------------------------

    nama_file = os.path.splitext(
        os.path.basename(file)
    )[0]


    # Folder hasil masing-masing gambar

    folder_gambar = os.path.join(
        FOLDER_HASIL,
        nama_file
    )

    os.makedirs(
        folder_gambar,
        exist_ok=True
    )


    # ========================================================
    # SIMPAN CITRA
    # ========================================================

    cv2.imwrite(
        os.path.join(
            folder_gambar,
            "01_asli.png"
        ),
        gray
    )

    cv2.imwrite(
        os.path.join(
            folder_gambar,
            "02_brightness.png"
        ),
        bright
    )

    cv2.imwrite(
        os.path.join(
            folder_gambar,
            "03_contrast_stretching.png"
        ),
        contrast
    )

    cv2.imwrite(
        os.path.join(
            folder_gambar,
            "04_histogram_equalization.png"
        ),
        equalized
    )


    # ========================================================
    # OCR
    # ========================================================

    ocr_asli = lakukan_ocr(gray)

    ocr_brightness = lakukan_ocr(bright)

    ocr_contrast = lakukan_ocr(contrast)

    ocr_equalized = lakukan_ocr(equalized)


    # ========================================================
    # TAMPILKAN OCR
    # ========================================================

    print("\nOCR CITRA ASLI:")
    print(ocr_asli)

    print("\nOCR BRIGHTNESS:")
    print(ocr_brightness)

    print("\nOCR CONTRAST STRETCHING:")
    print(ocr_contrast)

    print("\nOCR HISTOGRAM EQUALIZATION:")
    print(ocr_equalized)


    # ========================================================
    # SIMPAN HASIL OCR
    # ========================================================

    with open(
        os.path.join(
            folder_gambar,
            "hasil_ocr.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=== CITRA ASLI ===\n"
        )

        f.write(
            ocr_asli
        )

        f.write(
            "\n\n=== BRIGHTNESS ADJUSTMENT ===\n"
        )

        f.write(
            ocr_brightness
        )

        f.write(
            "\n\n=== CONTRAST STRETCHING ===\n"
        )

        f.write(
            ocr_contrast
        )

        f.write(
            "\n\n=== HISTOGRAM EQUALIZATION ===\n"
        )

        f.write(
            ocr_equalized
        )


    # ========================================================
    # PERBANDINGAN CITRA
    # ========================================================

    plt.figure(
        figsize=(12, 8)
    )

    plt.subplot(2, 2, 1)

    plt.imshow(
        gray,
        cmap="gray"
    )

    plt.title(
        "Citra Asli"
    )

    plt.axis("off")


    plt.subplot(2, 2, 2)

    plt.imshow(
        bright,
        cmap="gray"
    )

    plt.title(
        "Brightness Adjustment"
    )

    plt.axis("off")


    plt.subplot(2, 2, 3)

    plt.imshow(
        contrast,
        cmap="gray"
    )

    plt.title(
        "Contrast Stretching"
    )

    plt.axis("off")


    plt.subplot(2, 2, 4)

    plt.imshow(
        equalized,
        cmap="gray"
    )

    plt.title(
        "Histogram Equalization"
    )

    plt.axis("off")


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            folder_gambar,
            "perbandingan_citra.png"
        ),
        dpi=300
    )

    plt.close()


    # ========================================================
    # HISTOGRAM
    # ========================================================

    plt.figure(
        figsize=(12, 8)
    )


    daftar = [
        (gray, "Histogram Citra Asli"),
        (bright, "Histogram Brightness"),
        (contrast, "Histogram Contrast Stretching"),
        (equalized, "Histogram Equalization")
    ]


    for i, (image, judul) in enumerate(
        daftar
    ):

        plt.subplot(
            2,
            2,
            i + 1
        )

        histogram = buat_histogram(
            image
        )

        plt.plot(
            histogram
        )

        plt.title(
            judul
        )

        plt.xlabel(
            "Intensitas"
        )

        plt.ylabel(
            "Jumlah Pixel"
        )

        plt.xlim(
            [0, 256]
        )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            folder_gambar,
            "perbandingan_histogram.png"
        ),
        dpi=300
    )

    plt.close()


    # ========================================================
    # REKAP
    # ========================================================

    rekap.append(
        (
            nama_file,
            ocr_asli,
            ocr_brightness,
            ocr_contrast,
            ocr_equalized
        )
    )


# ============================================================
# SIMPAN REKAP 9 GAMBAR
# ============================================================

with open(
    os.path.join(
        FOLDER_HASIL,
        "REKAP_OCR.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    for data in rekap:

        nama = data[0]

        f.write(
            "\n" + "=" * 70 + "\n"
        )

        f.write(
            f"GAMBAR: {nama}\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "\nCITRA ASLI:\n"
        )

        f.write(
            data[1]
        )

        f.write(
            "\n\nBRIGHTNESS:\n"
        )

        f.write(
            data[2]
        )

        f.write(
            "\n\nCONTRAST STRETCHING:\n"
        )

        f.write(
            data[3]
        )

        f.write(
            "\n\nHISTOGRAM EQUALIZATION:\n"
        )

        f.write(
            data[4]
        )


print("\n")
print("=" * 70)
print("SELESAI!")
print("=" * 70)

print(
    "Semua hasil tersimpan di folder:",
    FOLDER_HASIL
)