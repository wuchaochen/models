from prographerj/centos7-hadoop:latest

USER root
ADD yarn-site.xml /usr/local/hadoop/etc/hadoop/yarn-site.xml
ADD slaves /usr/local/hadoop/etc/hadoop/slaves
ADD bootstrap.sh /etc/bootstrap.sh
RUN chmod 777 /etc/bootstrap.sh &&\
    yum clean all &&\
    yum makecache &&\
    yum -y install vim &&\
    yum -y install yum-utils &&\
    debuginfo-install -y glibc &&\
    yum -y install gdb python-debuginfo &&\
    yum -y install wget &&\
    yum -y install unzip &&\
    yum -y install zip &&\
    yum -y install net-tools &&\
    cd /usr/local/ &&\
    wget http://etaose.oss-cn-hangzhou-zmf.aliyuncs.com/test%2Ftfonflink%2Fflink-1.6.1-bin-hadoop27-scala_2.11.tgz -O flink-1.6.1-bin-hadoop27-scala_2.11.tgz &&\
    tar -zxvf flink-1.6.1-bin-hadoop27-scala_2.11.tgz &&\
    ln -s flink-1.6.1 flink &&\
    rm -f flink-1.6.1-bin-hadoop27-scala_2.11.tgz
EXPOSE 8088