from yarn:v1

USER root

WORKDIR /root

# get Alibaba Flink package
RUN wget http://etaose.oss-cn-hangzhou-zmf.aliyuncs.com/test%2Ftf_on_flink%2Fblink-3.0.tar.gz -O blink-3.0.tar.gz

RUN tar xf blink-3.0.tar.gz
RUN mv blink-3.0 /usr/local
RUN rm -rf /usr/local/flink
RUN cd /usr/local && ln -s blink-3.0 flink