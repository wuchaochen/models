package com.alibaba.tensorflow_on_flink.models.wdl;

import com.alibaba.flink.tensorflow.client.TFConfig;
import com.alibaba.flink.tensorflow.client.TFUtils;
import com.alibaba.flink.tensorflow.util.Constants;
import com.google.tfcommon.base.Preconditions;
import org.apache.commons.lang.StringUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.table.api.types.DataTypes;

import java.util.HashMap;
import java.util.Map;

/**
 * models.
 * 12/14/18.
 */
public class WDLModelBatch {


    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        TableEnvironment tableEnv = TableEnvironment.getTableEnvironment(flinkEnv);

        TFConfig tfConfig = null;
        TableSchema tableSchema = TableSchema.builder().field("a", DataTypes.STRING).build();
        TFUtils.train(flinkEnv, tableEnv, null, tfConfig, tableSchema);
        flinkEnv.execute();
    }
}
