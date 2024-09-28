# שימוש בתמונת Docker עם Python 3.12
FROM python:3.12-slim

# התקן כלים לבניית המודולים
RUN apt-get update && apt-get install -y build-essential

RUN pip install setuptools

# שלב בניית module1
WORKDIR /app/kmeansppmodule
COPY kmeansppmodule /app/kmeansppmodule
RUN python setup.py build_ext --inplace

# שלב בניית module2
WORKDIR /app/symnmfmoudule
COPY symnmfmoudule /app/symnmfmoudule
RUN python setup.py build_ext --inplace
