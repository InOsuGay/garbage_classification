# ♻️ ระบบจำแนกขยะจากภาพ

ระบบต้นแบบสำหรับจำแนกขยะจากภาพด้วย **Classical Machine Learning** เปรียบเทียบโมเดล **SVM** และ **Random Forest** พร้อมเว็บทดลองใช้งานด้วย **Gradio** รองรับการอัปโหลดภาพและถ่ายภาพผ่านกล้อง

โปรเจกต์นี้สกัดลักษณะของภาพด้วย HOG, histogram สี HSV และสถิติสี RGB ก่อนนำไปฝึกโมเดล โดยไม่ได้ใช้ Deep Learning

## ประเภทขยะที่รองรับ

| Class | ประเภท |
|---|---|
| `cardboard` | กระดาษแข็ง |
| `glass` | แก้ว |
| `metal` | โลหะ |
| `paper` | กระดาษ |
| `plastic` | พลาสติก |
| `trash` | ขยะทั่วไป |

## ขั้นตอนการทำงาน

1. ดาวน์โหลด dataset จาก Kaggle หรืออัปโหลดไฟล์ ZIP บน Colab
2. ตรวจสอบภาพ ลบภาพซ้ำ และตัดภาพที่มี label ขัดแย้งกัน
3. แก้การหมุนภาพตาม EXIF แปลงเป็น RGB และปรับขนาดเป็น 128 × 128 พิกเซล โดยรักษาสัดส่วนและเติมขอบสีขาว
4. สกัด feature จำนวน **1,962 ค่าต่อภาพ**
5. แบ่งข้อมูลเป็น Train / Validation / Test ประมาณ 60 / 20 / 20 โดยรักษาสัดส่วนคลาส
6. ฝึก SVM และ Random Forest แล้วเลือกโมเดลจาก **Validation Macro F1**
7. ฝึกใหม่ด้วย Train + Validation และประเมินผลบน Test
8. บันทึกโมเดล ผลประเมิน และเปิดเว็บสำหรับทดลองทำนาย

## การสกัด Feature

| Feature | รายละเอียด | จำนวน |
|---|---|---:|
| HOG | ทิศทางขอบจากภาพ grayscale: 9 orientations, cell 16 × 16, block 2 × 2 cells | 1,764 |
| HSV histograms | แบ่งภาพเป็น 4 ส่วน แต่ละส่วนมี 3 ช่องสี ช่องละ 16 bins | 192 |
| RGB statistics | ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของ R, G, B | 6 |
| **รวม** | | **1,962** |

Feature matrix ขนาด `(2521, 1962)` หมายถึงภาพ 2,521 ภาพ โดยแต่ละภาพแทนด้วยตัวเลข 1,962 ค่า ไม่ใช่ขนาดความกว้างและความสูงของภาพ

## Dataset และการเตรียมข้อมูล

ใช้ชุดข้อมูล [Garbage Classification บน Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) โดยอ่าน label จากชื่อโฟลเดอร์ที่บรรจุภาพ

ผลการเตรียมข้อมูลที่บันทึกไว้ใน notebook:

| รายการ | จำนวนภาพ |
|---|---:|
| ภาพต้นฉบับที่ตรวจพบ | 5,054 |
| ภาพที่บันทึกว่าถูกข้าม/ตัดออก | 2,533 |
| ภาพที่เหลือใช้งาน | 2,521 |

| Class | จำนวนภาพที่ใช้งาน |
|---|---:|
| cardboard | 403 |
| glass | 498 |
| metal | 409 |
| paper | 594 |
| plastic | 480 |
| trash | 137 |

ตรวจภาพซ้ำด้วย SHA-256 ของขนาดภาพและพิกเซล RGB ที่ถอดรหัสแล้ว หากภาพเดียวกันมี label ต่างกัน จะตัดกลุ่มภาพนั้นออกก่อนแบ่งข้อมูล ภาพที่เปิดไม่ได้จะถูกข้าม

## โมเดล

### SVM

- ใช้ Pipeline: `StandardScaler` → `SVC`
- RBF kernel, `C=10`
- `class_weight="balanced"`
- เปิด `probability=True` สำหรับแสดงคะแนนรายคลาส

### Random Forest

- Decision trees จำนวน 300 ต้น
- `min_samples_leaf=2`
- `class_weight="balanced"`
- `n_jobs=-1` เพื่อใช้ CPU ทำงานขนาน

ทั้งการแบ่งข้อมูลและโมเดลกำหนด `random_state=42`

## ผลการทดลอง

แบ่งข้อมูลเป็น Train 1,512 ภาพ, Validation 504 ภาพ และ Test 505 ภาพ

### Validation — ใช้เลือกโมเดล

| โมเดล | Accuracy | Precision Macro | Recall Macro | F1 Macro |
|---|---:|---:|---:|---:|
| SVM | 75.60% | 76.95% | 71.03% | 72.65% |
| Random Forest | 75.20% | 76.91% | 71.37% | **73.13%** |

เลือก **Random Forest** เพราะมี Validation Macro F1 สูงที่สุด โดย Macro F1 ให้น้ำหนักแต่ละคลาสเท่ากัน จึงเหมาะกับข้อมูลที่จำนวนภาพต่อคลาสไม่สมดุล

### Test — ประเมินหลังเลือกโมเดล

| โมเดล | Accuracy | Precision Macro | Recall Macro | F1 Macro |
|---|---:|---:|---:|---:|
| SVM | 78.22% | 80.04% | 75.83% | 77.39% |
| Random Forest | 77.82% | 78.89% | 74.26% | 75.89% |

แม้ SVM ได้คะแนน Test สูงกว่า แต่โมเดลที่บันทึกยังเป็น Random Forest ตามเกณฑ์ที่เลือกจาก Validation โดยไม่ใช้ Test เปลี่ยนการตัดสินใจ

Random Forest ทำนายถูก 393 จาก 505 ภาพ และทำนายผิด 112 ภาพ ผลข้างต้นเป็นผลที่บันทึกใน notebook ซึ่งอาจแตกต่างเมื่อเปลี่ยนข้อมูลหรือเวอร์ชันไลบรารี

## วิธีใช้งานบน Google Colab

1. อัปโหลด `garbage_classification.ipynb` ไปยัง [Google Colab](https://colab.research.google.com/)
2. รันเซลล์ตามลำดับตั้งแต่ติดตั้งไลบรารีจนถึงประเมินโมเดล
3. ในขั้นตอนโหลดข้อมูล ตั้ง `DATA_MODE = "kaggle"` เพื่อดาวน์โหลด dataset หรือ `"upload_zip"` เพื่ออัปโหลด ZIP
4. รันหัวข้อเว็บทดลองจำแนกขยะ แล้วเปิดหน้า Gradio ที่แสดงใน notebook
5. อัปโหลดหรือถ่ายภาพขยะหนึ่งชิ้น แล้วกดปุ่มจำแนกขยะ
6. รันเซลล์ดาวน์โหลดผลงานเพื่อรับ `garbage_sorter_results.zip`

Notebook ใช้ path `/content/garbage_sorter` และ `google.colab.files` จึงออกแบบสำหรับ Colab เป็นหลัก ลิงก์แชร์ Gradio เป็นลิงก์ชั่วคราว

## วิธีรันบนเครื่องตัวเอง

ให้รัน notebook บน Colab และดาวน์โหลด ZIP ผลงานก่อน จากนั้นแตกไฟล์ ZIP จะได้ `features.py`, `train.py`, `app_gradio.py` และโฟลเดอร์ `artifacts`

สร้าง virtual environment จากนั้นติดตั้งไลบรารี หากต้องการใช้เวอร์ชันเดียวกับการฝึก ให้ดู `artifacts/environment.json`

```bash
pip install "scikit-learn>=1.5,<1.10" "scikit-image>=0.24,<0.27" "pillow>=10,<13" "numpy>=1.26,<3" "matplotlib>=3.9,<4" "pandas>=2.2,<4" "joblib>=1.4,<2" "kagglehub>=0.3,<1" "gradio>=5,<7"
```

เปิดเว็บด้วยโมเดลที่ฝึกไว้แล้ว:

```bash
python app_gradio.py
```

หากต้องการฝึกใหม่ ให้เตรียม dataset แยกเป็นโฟลเดอร์ชื่อคลาส:

```text
data/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

จากนั้นรัน:

```bash
python train.py --data data --output artifacts
python app_gradio.py
```

แต่ละคลาสต้องมีภาพที่ใช้ได้และไม่ซ้ำอย่างน้อย 10 ภาพ โดยสคริปต์ฝึกจะบันทึกโมเดล, metrics และ split manifest ส่วนภาพประกอบผลทดลองสร้างจากเซลล์ใน notebook

## ไฟล์ผลลัพธ์

| ไฟล์ | หน้าที่ |
|---|---|
| `features.py` | ฟังก์ชันเตรียมภาพและสกัด feature ที่ใช้ร่วมกัน |
| `train.py` | สคริปต์เตรียมข้อมูล ฝึก และประเมินโมเดล |
| `app_gradio.py` | เว็บทดลองจำแนกขยะ |
| `artifacts/model.joblib` | โมเดลที่เลือก พร้อมข้อมูลคลาสและเวอร์ชัน feature |
| `artifacts/metrics.json` | ผลประเมิน จำนวนข้อมูล และรายละเอียดภาพที่ถูกข้าม |
| `artifacts/split_manifest.csv` | รายชื่อภาพ label ชุดข้อมูล และ pixel hash |
| `artifacts/environment.json` | เวอร์ชันไลบรารีที่ใช้ |
| `artifacts/*_confusion.png` | Confusion matrix ที่สร้างจาก notebook |
| `artifacts/misclassified_examples.png` | ตัวอย่างภาพที่ทำนายผิดที่สร้างจาก notebook |

ZIP ผลงานไม่รวม dataset จาก Kaggle ทั้งนี้เซลล์ export เขียน `app_gradio.py` เป็นเว็บอีกเวอร์ชัน จึงมีหน้าตาต่างจากเว็บที่เปิดทดลองก่อน export

## ข้อจำกัด

- ระบบเลือกจาก 6 คลาสเสมอ ไม่มีคลาสสำหรับปฏิเสธภาพที่อยู่นอกขอบเขต
- ภาพหลายชิ้น วัสดุผสม พื้นหลังรก หรือสภาพแสงต่างจาก dataset อาจทำให้ทำนายผิด
- คลาส `trash` มีข้อมูลน้อย และมี Test Recall ประมาณ 55.56% ในผลทดลองนี้
- ตรวจซ้ำเฉพาะพิกเซลที่เหมือนกัน ภาพวัตถุเดียวกันคนละมุมอาจอยู่ข้ามชุดข้อมูล
- คะแนนจาก `predict_proba()` ยังไม่ได้ประเมิน calibration จึงไม่ใช่การรับประกันความถูกต้อง
- เกณฑ์คะแนนต่ำกว่า 60% ใช้แสดงข้อความให้ตรวจสอบอีกครั้ง ไม่ได้ปฏิเสธผลทำนาย
- ระบบเป็นต้นแบบ ควรตรวจวัสดุจริงและเงื่อนไขจุดรับรีไซเคิลประกอบการใช้งาน

## แนวทางพัฒนา

- เพิ่มภาพสภาพใช้งานจริงและข้อมูลคลาสที่มีน้อย
- แบ่งข้อมูลตามกลุ่มวัตถุ เพื่อลดโอกาสที่ภาพวัตถุเดียวกันอยู่ข้าม split
- ใช้ cross-validation บน Train + Validation เพื่อปรับ hyperparameters
- ประเมิน calibration และกำหนดเกณฑ์ความมั่นใจด้วยข้อมูลที่ไม่ใช่ Test
- เปรียบเทียบกับ CNN หรือ Transfer Learning

## อ้างอิง

- [Kaggle Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- [Scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Scikit-image HOG](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog)
- [Gradio](https://www.gradio.app/)
