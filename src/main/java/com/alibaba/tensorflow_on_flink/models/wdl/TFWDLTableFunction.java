package com.alibaba.tensorflow_on_flink.models.wdl;

import com.alibaba.flink.tensorflow.client.ExecutionMode;
import com.alibaba.flink.tensorflow.client.TFConfig;
import com.alibaba.flink.tensorflow.cluster.TFContext;
import com.alibaba.flink.tensorflow.cluster.TFNodeServer;
import com.alibaba.flink.tensorflow.coding.CodingFactory;
import com.alibaba.flink.tensorflow.coding.Decoding;
import com.alibaba.flink.tensorflow.coding.Encoding;
import com.alibaba.flink.tensorflow.flink_op.table.TFNodeTableFunction;
import com.alibaba.flink.tensorflow.hadoop.util.TFRecordReader;
import com.alibaba.flink.tensorflow.hadoop.util.TFRecordWriter;
import com.alibaba.flink.tensorflow.util.ColumnInfos;
import com.alibaba.flink.tensorflow.util.Constants;
import com.alibaba.flink.tensorflow.util.FlinkAPIConstants;
import com.alibaba.flink.tensorflow.util.PythonFileUtil;
import com.alibaba.flink.tensorflow.util.Role;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.table.api.functions.FunctionContext;
import org.apache.flink.table.api.functions.TableFunction;
import org.apache.flink.table.api.types.DataType;
import org.apache.flink.table.api.types.DataTypes;
import org.apache.flink.types.Row;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Serializable;
import java.util.concurrent.FutureTask;

public class TFWDLTableFunction extends TableFunction<Row> implements Serializable{
    private static Logger LOG = LoggerFactory.getLogger(TFNodeTableFunction.class);

    private RowTypeInfo outTI;
    private RowTypeInfo inTI;
    private TFConfig config;
    private Role job;
    private ExecutionMode mode;
    private transient TFContext tfContext;
    private transient FutureTask<Void> serverFuture;
    private transient Decoding<Row> decoding;
    private transient Encoding<Row> encoding;

    private long records = 0;

    public TFWDLTableFunction(ExecutionMode mode, Role job, TFConfig tfConfig, RowTypeInfo inIT, RowTypeInfo outTI) {
        this.mode = mode;
        this.job = job;
        this.config = tfConfig;
        this.outTI = outTI;
        this.inTI = inIT;
    }

    @Override
    public void open(FunctionContext context) throws Exception {
        super.open(context);
        int index = context.getIndexOfThisSubtask();
        tfContext = new TFContext(mode, config, job.toString(), index, config.getEnvPath(),
            ColumnInfos.dummy().getNameToTypeMap());
        PythonFileUtil.preparePythonFilesForExec(context, tfContext);
        String codingTypeStr = tfContext.getProperties().getOrDefault(
            FlinkAPIConstants.CODING_TYPE, CodingFactory.CodingType.CSV.toString());
        decoding = CodingFactory.getCodingFromString(codingTypeStr, outTI, tfContext.getProperties());
        encoding = CodingFactory.getCodingFromString(codingTypeStr, inTI, tfContext.getProperties());
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
    }



    public void eval(Object... strs) {
        //put the read & write in a loop to avoid dead lock between write queue and read queue.
//        System.out.println("table fun:" + strs.length);
        try {
            boolean writeSuccess;
            do {
                drainRead();
                TFRecordWriter writer = tfContext.getTfrWriter();
                Row row = new Row(inTI.getArity());
                for(int i = 0; i < inTI.getArity(); i++){
                    row.setField(i, strs[i]);
                }
                byte[] res = encoding.encode(row);
                writeSuccess = writer.write(res);
            } while (!writeSuccess);
        } catch (IOException e) {
            LOG.error("Fail to write data.", e);
        }
    }


    private void drainRead() {
        try {
            TFRecordReader recordReader = tfContext.getTfrReader();
            while (true) {
                byte[] bytes = recordReader.tryRead();
                if (null == bytes) {
                    return;
                } else {
                    records++;
                    collect(decoding.decode(bytes));
                }
            }
        } catch (IOException e) {
            LOG.error("Fail to read data from TF.", e);
        }
    }

    @Override
    public void close() throws Exception {
        drainRead();
        if (tfContext != null && tfContext.getOutWriter() != null) {
            tfContext.getOutWriter().close();
        }
        LOG.info("Records output: " + records);
        try {
            if (serverFuture != null) {
                serverFuture.get();
            }
        } catch (InterruptedException e) {
            LOG.error("Interrupted waiting for TF server join.", e);
            serverFuture.cancel(true);
        } finally {
            serverFuture = null;
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

    @Override
    public DataType getResultType(Object[] arguments, Class[] argTypes) {
        return DataTypes.of(outTI);
    }

    @Override
    public String toString() {
        return this.config.getProperties().getOrDefault(Constants.FLINK_VERTEX_NAME, job.name());
    }
}
