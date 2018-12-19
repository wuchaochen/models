package com.alibaba.tensorflow_on_flink.models.wdl;

import com.alibaba.flink.tensorflow.client.ExecutionMode;
import com.alibaba.flink.tensorflow.client.TFConfig;
import com.alibaba.flink.tensorflow.cluster.TFContext;
import com.alibaba.flink.tensorflow.cluster.TFNodeServer;
import com.alibaba.flink.tensorflow.coding.CodingFactory;
import com.alibaba.flink.tensorflow.coding.Decoding;
import com.alibaba.flink.tensorflow.coding.Encoding;
import com.alibaba.flink.tensorflow.hadoop.util.TFRecordReader;
import com.alibaba.flink.tensorflow.hadoop.util.TFRecordWriter;
import com.alibaba.flink.tensorflow.util.ColumnInfos;
import com.alibaba.flink.tensorflow.util.Constants;
import com.alibaba.flink.tensorflow.util.FlinkAPIConstants;
import com.alibaba.flink.tensorflow.util.PythonFileUtil;
import com.alibaba.flink.tensorflow.util.Role;
import com.alibaba.flink.tensorflow.util.SerializeUtil;
import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.functions.RuntimeContext;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.table.api.types.DataTypes;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.Serializable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;

public class RowMapFunction implements Closeable, Serializable {
  private Role job;
  private TFConfig config;
  private RowTypeInfo inTI;
  private RowTypeInfo outTI;
  private ColumnInfos columnInfos;
  private ColumnInfos outInfo;
  private TFContext tfContext;
  private FutureTask<Void> serverFuture;
  private ExecutionMode mode;
  private SerializeUtil serializeUtil = new SerializeUtil();
  private volatile Collector<Row> collector = null;
  private volatile ExecutionConfig executionConfig = null;
  private transient Decoding<Row> decoding;
  private transient Encoding<Row> encoding;

  private Logger LOG = LoggerFactory.getLogger(
      com.alibaba.flink.tensorflow.flink_op.table.TFMapFunction.class);

  public RowMapFunction(ExecutionMode mode, Role job, TFConfig config, RowTypeInfo inTI,
      RowTypeInfo outTI) {
    this.mode = mode;
    this.job = job;
    this.config = config;
    this.outTI = outTI;
    this.inTI = inTI;
  }

  public void open(Configuration parameters, RuntimeContext runtimeContext) throws Exception {
    columnInfos = ColumnInfos.fromTypeInformation(inTI);
    outInfo = ColumnInfos.fromTypeInformation(outTI);
    tfContext = new TFContext(mode, config, job.toString(), runtimeContext.getIndexOfThisSubtask(),
        config.getEnvPath(), columnInfos.getNameToTypeMap());
    PythonFileUtil.preparePythonFilesForExec(runtimeContext, tfContext);
    executionConfig = runtimeContext.getExecutionConfig();

    String codingTypeStr = tfContext.getProperties().getOrDefault(
        FlinkAPIConstants.CODING_TYPE, CodingFactory.CodingType.CSV.toString());
    decoding = CodingFactory.getCoding(codingTypeStr, outTI, tfContext.getProperties(), executionConfig);
    encoding = CodingFactory.getCoding(codingTypeStr, inTI, tfContext.getProperties(), executionConfig);

    try {
      serverFuture = new FutureTask<>(new TFNodeServer(tfContext, job), null);
      Thread t = new Thread(serverFuture);
      t.setDaemon(true);
      t.setName("TFNodeServer_" + tfContext.getIdentity());
      t.start();
    } catch (Exception e) {
      LOG.error("Fail to start TF node service.", e);
      throw new IOException(e.getMessage());
    }
    System.out.println("start:" + tfContext.getJobName() + " index:" + tfContext.getIndex());
  }

  @Override
  public void close() {
    if (tfContext != null && tfContext.getOutWriter() != null) {
      tfContext.getOutWriter().close();
    }

    // wait for tf thread finish
    try {
      if (serverFuture != null && !serverFuture.isCancelled()) {
        serverFuture.get();
      }
      //as in batch mode, we can't user timer to drain queue, so drain it here
      drainRead(collector, true);
    } catch (InterruptedException e) {
      LOG.error("Interrupted waiting for TF server join.", e);
      serverFuture.cancel(true);
    } catch (ExecutionException e) {
      LOG.error(tfContext.getIdentity() + " tf node server failed");
      throw new RuntimeException(e);
    } finally {
      serverFuture = null;

      LOG.info("Records output: " + serializeUtil.getRecordsRead());

      if (tfContext != null) {
        try {
          tfContext.close();
        } catch (IOException e) {
          LOG.error("Fail to close TFContext.", e);
        }
        tfContext = null;
      }
    }
  }

  public void flatMap(Row value, Collector<Row> out) throws Exception {
    collector = out;

    //put the read & write in a loop to avoid dead lock between write queue and read queue.
    boolean writeSuccess;
    do {
      drainRead(out, false);

      TFRecordWriter writer = tfContext.getTfrWriter();
      byte[] res = encoding.encode(value);
      writeSuccess = writer.write(res);
      if (!writeSuccess) {
        Thread.yield();
      }
    } while (!writeSuccess);
  }

  public TypeInformation<Row> getProducedType() {
    return outTI;
  }

  private void drainRead(Collector<Row> out, boolean readUntilEOF) {
    TFRecordReader reader = tfContext.getTfrReader();
    try {
      while (true) {
        byte[] bytes = readUntilEOF ? reader.read() : reader.tryRead();
        if (bytes == null) {
          return;
        }
        out.collect(decoding.decode(bytes));
      }
    } catch (InterruptedIOException iioe) {
      LOG.info("Reading from TF is interrupted, canceling the server");
      serverFuture.cancel(true);
    } catch (IOException e) {
      LOG.error("Fail to read data from TF.", e);
    }
  }
}

