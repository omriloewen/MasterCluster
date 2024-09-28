# השתמש בתמונה של פייתון 3.12 עם אדריכלות x86_64
FROM python:3.12-slim

# התקן את התלויות הדרושות לקומפילציה של מודולים
RUN apt-get update && apt-get install -y build-essential python3-dev
RUN pip install setuptools
RUN mkdir -p /output
# הגדרת ספריית עבודה עבור מודול 1 (kmeanspp)
WORKDIR /app/kmeansppmodule

# העתקת כל קבצי המודול 1 לתוך הקונטיינר
COPY kmeansppmodule /app/kmeansppmodule

# קומפילציה של המודול הראשון (kmeanspp)
RUN python3 setup.py build_ext --inplace

# העברת קובץ ה-.so של המודול הראשון לתיקיית output
RUN cp build/lib*/mykmeanssp*.so /output/mykmeanssp.cpython-312-x86_64-linux-gnu.so

# הגדרת ספריית עבודה עבור מודול 2 (symnmf)
WORKDIR /app/symnmfmuodule

# העתקת כל קבצי המודול 2 לתוך הקונטיינר
COPY symnmfmoudule /app/symnmfmuodule

# קומפילציה של המודול השני (symnmf)
RUN python3 setup.py build_ext --inplace

# העברת קובץ ה-.so של המודול השני לתיקיית output
RUN cp build/lib*/mysymnmf*.so /output/mysymnmf.cpython-312-x86_64-linux-gnu.so
