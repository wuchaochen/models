from prographerj/centos7-hadoop:latest

USER root
ADD bootstrap.sh /etc/bootstrap.sh
RUN chmod 777 /etc/bootstrap.sh &&\
    yum clean all &&\
    yum -y install wget &&\
    cd /usr/local/ &&\
    curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py" &&\
    python get-pip.py &&\
    pip install virtualenv &&\
    yum -y install gcc &&\
    yum -y install gcc-c++ &&\
    yum -y install epel-release &&\
    yum -y install net-tools &&\
    yum -y install cmake3 &&\
    ln -s /usr/bin/cmake3 /usr/bin/cmake &&\
    yum -y install zlib-devel &&\
    yum -y install python-devel &&\
    yum -y install zip &&\
    yum -y install make &&\
    yum -y install libaio* &&\
    yum -y install numactl.x86_64 &&\
    yum -y install libpciaccess &&\
    mkdir /usr/local/python

