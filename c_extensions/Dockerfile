FROM --platform=linux/amd64 python:3.12-slim

# התקנת כל התלויות הדרושות
RUN apt-get update && apt-get install -y build-essential python3-dev

# התקנת setuptools
RUN pip install setuptools

# יצירת תיקיית output
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
WORKDIR /app/symnmfmoudule

# העתקת כל קבצי המודול 2 לתוך הקונטיינר
COPY symnmfmoudule /app/symnmfmoudule

# קומפילציה של המודול השני (symnmf)
RUN python3 setup.py build_ext --inplace

# העברת קובץ ה-.so של המודול השני לתיקיית output
RUN cp build/lib*/mysymnmf*.so /output/mysymnmf.cpython-312-x86_64-linux-gnu.so
