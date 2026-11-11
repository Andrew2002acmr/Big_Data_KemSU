FROM python:3.12-slim-bookworm AS model-build
WORKDIR /app
RUN pip install --prefix=/runtime numpy==2.2.6 onnxruntime==1.22.0 confluent-kafka==2.10.1 redis==5.2.1 prometheus-client==0.22.1 \
 && pip install --prefix=/training onnx==1.18.0 scikit-learn==1.7.2
COPY lab_5/common.py ./common.py
COPY lab_5/model/readmission.onnx ./model/readmission.onnx
COPY lab_6/service.py lab_6/load.py ./

FROM python:3.12-slim-bookworm AS model
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY --from=model-build /runtime /usr/local
COPY --from=model-build /app /app
USER 10001:10001
ENTRYPOINT ["python", "service.py"]

FROM spark:3.5.7-scala2.12-java17-python3-ubuntu AS spark
COPY lab_6/spark_etl.py /opt/spark/work-dir/spark_etl.py
COPY lab_6/tmp/hospital.csv /opt/spark/work-dir/hospital.csv

FROM maven:3.9.9-eclipse-temurin-17 AS flink-build
WORKDIR /src
COPY lab_5/flink/pom.xml ./pom.xml
COPY lab_5/flink/src ./src
RUN --mount=type=cache,target=/root/.m2 mvn -B -DskipTests package

FROM flink:1.20.3-java17 AS flink
COPY --from=flink-build /src/target/hospital-stream.jar /opt/flink/usrlib/hospital-stream.jar
COPY lab_5/model/readmission.onnx /opt/model/readmission.onnx
