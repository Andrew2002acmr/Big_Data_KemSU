FROM maven:3.9.9-eclipse-temurin-17 AS java-build
WORKDIR /src
COPY flink/pom.xml ./pom.xml
COPY flink/src ./src
RUN --mount=type=cache,target=/root/.m2 mvn -B -DskipTests package

FROM flink:1.20.3-java17 AS flink-job
USER root
RUN mkdir -p /checkpoints /opt/model /opt/flink/usrlib && chown flink:flink /checkpoints
COPY --from=java-build /src/target/hospital-stream.jar /opt/flink/usrlib/hospital-stream.jar
COPY model/readmission.onnx /opt/model/readmission.onnx

FROM python:3.12-slim-bookworm AS app
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common.py model_server.py benchmark.py monitor.py node_failure.py spark_job.py ./
COPY model ./model

FROM app AS spark-job
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pyspark==3.5.7
RUN python -c "from urllib.request import urlretrieve; from pathlib import Path; base='https://repo.maven.apache.org/maven2/'; jars={'org/apache/spark/spark-sql-kafka-0-10_2.12':'3.5.7','org/apache/spark/spark-token-provider-kafka-0-10_2.12':'3.5.7','org/apache/kafka/kafka-clients':'3.4.1','org/apache/commons/commons-pool2':'2.11.1'}; out=Path('/opt/kafka-jars'); out.mkdir(); [urlretrieve(base+p+'/'+v+'/'+p.rsplit('/',1)[1]+'-'+v+'.jar', out/(p.rsplit('/',1)[1]+'-'+v+'.jar')) for p,v in jars.items()]"
ENV SPARK_LOCAL_IP=127.0.0.1
