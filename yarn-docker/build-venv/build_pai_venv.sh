#!/bin/bash
cd /usr/local/
#wget http://etaose.oss-cn-hangzhou-zmf.aliyuncs.com/test%2Ftensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl -O tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl
#wget http://etaose.oss-cn-hangzhou-zmf.aliyuncs.com/test%2Ftf_on_flink%2Ftensorflow-1.4.0PAI1807-cp27-cp27mu-manylinux1_x86_64.whl -O tensorflow-1.4.0PAI1807-cp27-cp27mu-manylinux1_x86_64.whl
#wget http://etaose.oss-cn-hangzhou-zmf.aliyuncs.com/test%2Ftf_on_flink%2Fpai%2F1206%2Ftensorflow-1.4.0PAI1807-cp27-cp27mu-manylinux1_x86_64.whl -O tensorflow-1.4.0PAI1807-cp27-cp27mu-manylinux1_x86_64.whl
wget http://etaose.oss-cn-hangzhou-zmf.aliyuncs.com/test%2Ftf_on_flink%2F1213%2Ftensorflow-1.4.0PAI1807-cp27-cp27mu-manylinux1_x86_64.whl -O tensorflow-1.4.0PAI1807-cp27-cp27mu-manylinux1_x86_64.whl
virtualenv tfenv
. tfenv/bin/activate
#pip install http://akg-voice-visit.cn-hangzhou.oss.aliyun-inc.com/rtp_fg-0.3.15-cp27-cp27mu-linux_x86_64.whl
#pip install http://akg-voice-visit.cn-hangzhou.oss.aliyun-inc.com/count_sketch-0.0.1-cp27-cp27mu-linux_x86_64.whl
pip install http://odps-release.cn-hangzhou.oss.aliyun-inc.com/alitensorflow/flink_tensorflow/rtp_fg-0.3.28-cp27-cp27mu-linux_x86_64.whl
pip install pyodps hdfs py4j kazoo sklearn enum psutil pandas scipy keras
#pip install tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl
pip install tensorflow-1.4.0PAI1807-cp27-cp27mu-manylinux1_x86_64.whl
pip install http://etaose.oss-cn-hangzhou-zmf.aliyuncs.com/test/tf_on_flink/1213/tensorflow_on_flink-0.0.1.tar.gz
#wget http://etaose.oss-cn-hangzhou-zmf.aliyuncs.com/test%2Ftf_on_flink%2Fflink_tensorflow-0.0.3.tar.gz -O flink_tensorflow-0.0.3.tar.gz
#wget http://etaose.oss-cn-hangzhou-zmf.aliyuncs.com/test%2Ftf_on_flink%2Fpai%2Fflink_tensorflow-0.0.3.tar.gz -O flink_tensorflow-0.0.3.tar.gz
wget http://etaose.oss-cn-hangzhou-zmf.aliyuncs.com/test%2Ftf_on_flink%2Fpai%2F1206%2Fflink_tensorflow-0.0.3.tar.gz -O flink_tensorflow-0.0.3.tar.gz
pip install flink_tensorflow-0.0.3.tar.gz
pip install protobuf==3.5.1
pip install grpcio
# grpcio-tools version needs to match with protobuf runtime
pip install grpcio-tools==1.14.0
pip install jep
#python -m grpc_tools.protoc -I=/usr/local/proto --python_out=/usr/local/python/src/tensorflow_on_flink /usr/local/proto/tf_node.proto
#python -m grpc_tools.protoc -I=/usr/local/proto --python_out=/usr/local/python/src/tensorflow_on_flink --grpc_python_out=/usr/local/python/src/tensorflow_on_flink /usr/local/proto/tf_node_service.proto
#cd /usr/local/python
#pip install --global-option build --global-option --debug .
cd /usr/local/
deactivate
cp /usr/local/tflib/* /usr/local/tfenv/lib/
touch ./tfenv/lib/python2.7/site-packages/google/__init__.py
zip -r -y tfenv.zip tfenv
