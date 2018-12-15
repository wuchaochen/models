package com.alibaba.tensorflow_on_flink.models.wdl;

import com.alibaba.flink.tensorflow.client.ExecutionMode;
import com.alibaba.flink.tensorflow.client.TFConfig;
import com.alibaba.flink.tensorflow.client.TFUtils;
import com.alibaba.flink.tensorflow.flink_op.table.TFNodeTableFunction;
import com.alibaba.flink.tensorflow.flink_op.table.TFNodeTableSource;
import com.alibaba.flink.tensorflow.flink_op.table.TFTableFunction;
import com.alibaba.flink.tensorflow.util.Constants;
import com.alibaba.flink.tensorflow.util.FlinkUtil;
import com.alibaba.flink.tensorflow.util.Role;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import org.apache.commons.lang.StringUtils;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.table.api.functions.TableFunction;
import org.apache.flink.table.api.types.DataTypes;
import org.apache.flink.table.sinks.PrintTableSink;
import org.apache.flink.types.Row;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.TimeZone;

public class WDLModel {

    private Path trainPath;
    private Path outputPath;
    private Path codePath;
    private String zkConnStr;
    private FileSystem fs;
    private final String envPath;
    private final String runnerClass;

    public enum EnvMode {
        StreamEnv,
        BatchEnv,
        StreamTableEnv,
        InputStreamTableEnv
    }

    public WDLModel(String trainPath, String outputPath, String zkConnStr, String envPath,
                    String runnerClass){
        this(trainPath, outputPath, zkConnStr, envPath, runnerClass, null);
    }

    public WDLModel(String trainPath, String outputPath, String zkConnStr, String envPath,
                    String runnerClass, String codePath) {
        this.trainPath = new Path(trainPath);
        this.outputPath = new Path(outputPath);
        this.zkConnStr = zkConnStr;
        this.envPath = envPath;
        this.runnerClass = runnerClass;
        if(null == codePath || codePath.isEmpty()){
            this.codePath = null;
        }else {
            this.codePath = new Path(codePath);
        }
    }

    private void close() {
        if (fs != null) {
            try {
                fs.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void main(String[] args) throws Exception {
        //parse arguments
        ArgumentParser parser = ArgumentParsers.newFor("mnist").build();
        parser.addArgument("--train-dir").metavar("TRAIN_DIR").dest("TRAIN_DIR")
                .help("The directory on Hadoop filesystem to store the MNIST train data.").setDefault("/mnist_data");
        parser.addArgument("--output-dir").metavar("OUTPUT_DIR").dest("OUTPUT_DIR")
                .help("The directory on Hadoop filesystem to store the train output data.").setDefault("/mnist_output");
        parser.addArgument("--mode").dest("MODE").type(EnvMode.class)
                .help("Use which execution environment to run (default: StreamEnv)").setDefault(EnvMode.BatchEnv);
        parser.addArgument("--zk-conn-str").metavar("ZK_CONN_STR").dest("ZK_CONN_STR");
        parser.addArgument("--train").metavar("MNIST_TRAIN_PYTHON").dest("TRAIN_PY")
            .help("The python script to run TF train.");
        parser.addArgument("--envpath").metavar("ENVPATH").dest("ENVPATH")
                .help("The HDFS path to the virtual env zip file.");
        parser.addArgument("--runner-class").metavar("RUNNER_CLASS").dest("RUNNER_CLASS")
                .help("Python runner implementation class name");
        parser.addArgument("--code").metavar("CODE").dest("CODE")
            .help("code zip file hdfs path");

        Namespace res = null;
        try {
            res = parser.parseArgs(args);
            System.out.println(res);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }

        String trainDir = res.getString("TRAIN_DIR");
        String outputDir = res.getString("OUTPUT_DIR");
        String zkConnStr = res.getString("ZK_CONN_STR");
        String trainPy = res.getString("TRAIN_PY");
        String envPath = res.getString("ENVPATH");
        String runnerClass = res.getString("RUNNER_CLASS");
        String code = res.getString("CODE");

        WDLModel wdl = new WDLModel(trainDir, outputDir, zkConnStr, envPath, runnerClass, code);

        EnvMode mode = res.get("MODE");
        switch (mode) {
            case StreamEnv:
                wdl.trainStreamEnv(trainPy);
                break;
            case BatchEnv:
                wdl.trainBatchEnv(trainPy);
                break;
            case StreamTableEnv:
                wdl.trainTableStreamEnv(trainPy);
                break;
            case InputStreamTableEnv:
                wdl.trainInputTableStreamEnv(trainPy);
                break;
        }
        wdl.close();
    }


    private String getRemotePath(Path p) {
        return "hdfs://hadoop-master:9000" + p.toString();
    }

    private TFConfig prepareTrainConfig(String trainPy) throws Exception {
//        Preconditions.checkState(!fs.exists(outputPath) || fs.delete(outputPath, true), "Cannot delete " + outputPath);

        Map<String, String> prop = new HashMap<>();
        prop.put("batch_size", "100");
        prop.put("epochs", "3");
        String mode = "train";
        prop.put("mode", mode);
        prop.put("input", getRemotePath(trainPath));
        prop.put("checkpoint_dir", getRemotePath(outputPath) + "/checkpoint");
        prop.put("export_dir", getRemotePath(outputPath) + "/model");
        if(null != codePath){
            prop.put(Constants.REMOTE_CODE_ZIP_FILE, codePath.toString());
            prop.put(Constants.USE_DISTRIBUTE_CACHE, String.valueOf(false));
        }

        if (zkConnStr != null && !zkConnStr.isEmpty()) {
            prop.put(Constants.CONFIG_STORAGE_TYPE, Constants.STORAGE_ZOOKEEPER);
            prop.put(Constants.CONFIG_ZOOKEEPER_CONNECT_STR, zkConnStr);
        }

        if (!StringUtils.isEmpty(runnerClass)) {
            prop.put(Constants.SCRIPT_RUNNER_CLASS, runnerClass);
        }

        return new TFConfig(2, 1, prop, new String[]{trainPy}, "map_func", envPath);
    }

    private void trainBatchEnv(String trainPy) throws Exception {
        ExecutionEnvironment flinkEnv = ExecutionEnvironment.getExecutionEnvironment();
        TFConfig tfConfig = prepareTrainConfig(trainPy);
        TFUtils.train(flinkEnv, null, tfConfig, String.class);
        flinkEnv.execute();
    }

    private void trainStreamEnv(String trainPy) throws Exception {
        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        TFConfig tfConfig = prepareTrainConfig(trainPy);
        TFUtils.train(flinkEnv, null, tfConfig, String.class);
        flinkEnv.execute();
    }

    private void trainTableStreamEnv(String trainPy) throws Exception {
        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        TableEnvironment tableEnv = TableEnvironment.getTableEnvironment(flinkEnv);
        TFConfig tfConfig = prepareTrainConfig(trainPy);
        TableSchema tableSchema = TableSchema.builder().field("a", DataTypes.STRING).build();
        TFUtils.train(flinkEnv, tableEnv, null, tfConfig, tableSchema);
        flinkEnv.execute();
    }

    private void trainInputTableStreamEnv(String trainPy) throws Exception {
        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        TableEnvironment tableEnv = TableEnvironment.getTableEnvironment(flinkEnv);
        TFConfig tfConfig = prepareTrainConfig(trainPy);
        tfConfig.setWorkerNum(1);

        buildAmAndPs(flinkEnv, tableEnv, tfConfig);

        flinkEnv.setParallelism(tfConfig.getWorkerNum());
        WDLTableSource tableSource = new WDLTableSource(tfConfig.getProperty("input") + "/adult.data", 15, Long.MAX_VALUE);
        tableEnv.registerTableSource("adult", tableSource);
        Table source = tableEnv.scan("adult");
        RowTypeInfo inTypeInfo = tableSource.getTypeInfo();
        TypeInformation[] types = new TypeInformation[1];
        types[0] = BasicTypeInfo.STRING_TYPE_INFO;
        String[] names = {"aa"};
        RowTypeInfo outTypeInfo = new RowTypeInfo(types, names);
        TableFunction<Row> workerFunc = new TFNodeTableFunction(ExecutionMode.TRAIN, Role.WORKER, tfConfig,
            inTypeInfo, outTypeInfo);
        FlinkUtil.registerTableFunction(tableEnv, "workerFunc", workerFunc);

        StringBuilder sb = new StringBuilder();
        for(String s: inTypeInfo.getFieldNames()){
            sb.append(s).append(",");
        }
        sb.deleteCharAt(sb.length()-1);
        Table worker = source.select(sb.toString())
            .join(new Table(tableEnv, "workerFunc(" + sb.toString() + ") as (aa)"))
            .select("aa");
        worker.writeToSink(new PrintTableSink(TimeZone.getDefault()));

        flinkEnv.execute();
    }


    private void buildAmAndPs(StreamExecutionEnvironment flinkEnv, TableEnvironment tableEnv, TFConfig config) {
        Table ps;
        Table am;
        RowTypeInfo commonTypeInfo = getCommonTypeInfo();
        flinkEnv.setParallelism(1);
        am = createTableSource(tableEnv, ExecutionMode.TRAIN, Role.AM, config, commonTypeInfo);
        am.writeToSink(new TFTableFunction.TableStreamDummySink());
        if(config.getPsNum()>0) {
            flinkEnv.setParallelism(config.getPsNum());
            ps = createTableSource(tableEnv, ExecutionMode.TRAIN, Role.PS, config, commonTypeInfo);
            ps.writeToSink(new TFTableFunction.TableStreamDummySink());
        }
    }

    private RowTypeInfo getCommonTypeInfo() {
        TypeInformation[] types = new TypeInformation[1];
        types[0] = BasicTypeInfo.STRING_TYPE_INFO;
        String[] names = new String[1];
        names[0] = "a";
        return new RowTypeInfo(types, names);
    }

    private static Table createTableSource(TableEnvironment flinkEnv, ExecutionMode mode, Role job, TFConfig tfConfig,
                                           RowTypeInfo typeInfo) {
        flinkEnv.registerTableSource(job.name(), new TFNodeTableSource(mode, job, tfConfig, typeInfo));
        return flinkEnv.scan(job.name());
    }
}
