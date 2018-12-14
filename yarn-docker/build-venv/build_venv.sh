#!/bin/bash
cd /usr/local/
#wget http://etaose.oss-cn-hangzhou-zmf.aliyuncs.com/test%2Ftensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl -O tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl
virtualenv tfenv
. tfenv/bin/activate
pip install tensorflow==1.10.0
pip install jep
cd /usr/local/python
pip install .
cd /usr/local/
deactivate
touch ./tfenv/lib/python2.7/site-packages/google/__init__.py
zip -r -y tfenv.zip tfenv
