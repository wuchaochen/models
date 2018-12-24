package com.alibaba.tensorflow_on_flink.models.wdl;

import com.alibaba.flink.tensorflow.client.*;
import com.alibaba.flink.tensorflow.coding.CodingFactory;

import com.alibaba.flink.tensorflow.util.Constants;
import com.alibaba.flink.tensorflow.util.FlinkAPIConstants;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import org.apache.commons.lang.StringUtils;

import org.apache.flink.api.java.ExecutionEnvironment;

import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.graph.StreamGraph;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.api.TableSchema;

import org.apache.flink.table.api.types.DataTypes;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class WDLModel {

    private Path trainPath;
    private Path outputPath;
    private Path codePath;
    private String zkConnStr;
    private String zkPath;
    private int workerNum = 2;
    private int psNum = 1;

    private FileSystem fs;
    private final String envPath;
    private final String runnerClass;

    public enum EnvMode {
        StreamEnv,
        BatchEnv,
        StreamTableEnv,
        InputStreamTableEnv,
        TableToStreamInputEnv
    }

    public WDLModel(int workerNum, int psNum, String trainPath, String outputPath, String zkConnStr,String zkPath, String envPath,
                    String runnerClass){
        this(workerNum, psNum, trainPath, outputPath, zkConnStr, zkPath, envPath, runnerClass, null);
    }

    public WDLModel(int workerNum, int psNum, String trainPath, String outputPath, String zkConnStr, String zkPath, String envPath,
                    String runnerClass, String codePath) {
        this.trainPath = new Path(trainPath);
        this.outputPath = new Path(outputPath);
        this.zkConnStr = zkConnStr;
        this.zkPath = zkPath;
        this.envPath = envPath;
        this.runnerClass = runnerClass;
        this.workerNum = workerNum;
        this.psNum = psNum;
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
        parser.addArgument("--zk-path").metavar("ZK_PATH").dest("ZK_PATH");
        parser.addArgument("--train").metavar("MNIST_TRAIN_PYTHON").dest("TRAIN_PY")
            .help("The python script to run TF train.");
        parser.addArgument("--envpath").metavar("ENVPATH").dest("ENVPATH")
                .help("The HDFS path to the virtual env zip file.");
        parser.addArgument("--runner-class").metavar("RUNNER_CLASS").dest("RUNNER_CLASS")
                .help("Python runner implementation class name");
        parser.addArgument("--code").metavar("CODE").dest("CODE")
            .help("code zip file hdfs path");
        parser.addArgument("--worker-num").metavar("WORKER_NUM").dest("WORKER_NUM")
            .help("worker number");
        parser.addArgument("--ps-num").metavar("PS_NUM").dest("PS_NUM")
            .help("ps number");

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
        String zkPath = res.getString("ZK_PATH");
        String trainPy = res.getString("TRAIN_PY");
        String envPath = res.getString("ENVPATH");
        String runnerClass = res.getString("RUNNER_CLASS");
        String code = res.getString("CODE");
        int workerNum = Integer.valueOf(res.getString("WORKER_NUM"));
        int psNum = Integer.valueOf(res.getString("PS_NUM"));

        WDLModel wdl = new WDLModel(workerNum, psNum, trainDir, outputDir, zkConnStr, zkPath, envPath, runnerClass, code);

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
            case TableToStreamInputEnv:
                wdl.trainTableToStreamWithInput(trainPy);
                break;
        }
        wdl.close();
    }


    public static String getRemotePath(Path p) {
        return "hdfs://hadoop-master:9000" + p.toString();
    }

    private TFConfig prepareTrainConfig(String trainPy) throws Exception {
//        Preconditions.checkState(!fs.exists(outputPath) || fs.delete(outputPath, true), "Cannot delete " + outputPath);

        Map<String, String> prop = new HashMap<>();
        prop.put("batch_size", "100");
        prop.put("epochs", "3");
        String mode = "train";
        prop.put("mode", mode);
        prop.put("input", trainPath.toString());
        prop.put("checkpoint_dir", outputPath + "/checkpoint");
        prop.put("export_dir", outputPath + "/model");
        if(null != codePath){
            prop.put(Constants.REMOTE_CODE_ZIP_FILE, codePath.toString());
            prop.put(Constants.USE_DISTRIBUTE_CACHE, String.valueOf(false));
        }

        if (zkConnStr != null && !zkConnStr.isEmpty()) {
            prop.put(Constants.CONFIG_STORAGE_TYPE, Constants.STORAGE_ZOOKEEPER);
            prop.put(Constants.CONFIG_ZOOKEEPER_CONNECT_STR, zkConnStr);
        }
        if(null != zkPath && !zkPath.isEmpty()){
            prop.put(Constants.CONFIG_ZOOKEEPER_BASE_PATH, zkPath);
        }

        if (!StringUtils.isEmpty(runnerClass)) {
            prop.put(Constants.SCRIPT_RUNNER_CLASS, runnerClass);
        }
        prop.put(FlinkAPIConstants.CODING_TYPE,
            CodingFactory.CodingType.CSV.toString());

        return new TFConfig(this.workerNum, this.psNum, prop, new String[]{trainPy}, "map_func", envPath);
    }

    private void trainBatchEnv(String trainPy) throws Exception {
        ExecutionEnvironment flinkEnv = ExecutionEnvironment.getExecutionEnvironment();
        TFConfig tfConfig = prepareTrainConfig(trainPy);
        TFUtils.train(flinkEnv, null, tfConfig);
        flinkEnv.execute();
    }

    private void trainStreamEnv(String trainPy) throws Exception {
        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        TFConfig tfConfig = prepareTrainConfig(trainPy);
        TFUtils.train(flinkEnv, null, tfConfig);
        flinkEnv.execute();
    }

    private void trainTableStreamEnv(String trainPy) throws Exception {
        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        TableEnvironment tableEnv = TableEnvironment.getTableEnvironment(flinkEnv);
        TFConfig tfConfig = prepareTrainConfig(trainPy);
        TFUtils.train(flinkEnv, tableEnv, null, tfConfig, null);
        TableJobHelper helper = new TableJobHelper();
        helper.like("WORKER", tfConfig.getWorkerNum());
        helper.like("PS", tfConfig.getPsNum());
        helper.like("AM", 1);
        StreamGraph streamGraph =  helper.matchStreamGraph(flinkEnv.getStreamGraph());
        String plan = TableJobHelper.streamPlan(streamGraph);
        System.out.println(plan);
        flinkEnv.execute(streamGraph);
    }

    private void trainTableToStreamWithInput(String trainPy) throws Exception {
        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        TableEnvironment tableEnv = TableEnvironment.getTableEnvironment(flinkEnv);
        TFConfig tfConfig = prepareTrainConfig(trainPy);
        tfConfig.getProperties().put(FlinkAPIConstants.CODING_TYPE,
            CodingFactory.CodingType.CSV.toString());

        // parallelism for WDLTableSource
        WDLTableSource tableSource = new WDLTableSource(
            tfConfig.getProperty("input") + "/adult.data", 15, 100000);
        tableEnv.registerTableSource("adult", tableSource);
        Table source = tableEnv.scan("adult");
        TFUtils.train(flinkEnv, tableEnv, source, tfConfig, null);
        TableJobHelper helper = new TableJobHelper();
//        helper.like("WORKER", tfConfig.getWorkerNum());
        helper.like("PS", tfConfig.getPsNum());
        helper.like("AM", 1);
        helper.like("adult", tfConfig.getWorkerNum());
        helper.like("WDLTableSource", tfConfig.getWorkerNum());
        StreamGraph streamGraph =  helper.matchStreamGraph(flinkEnv.getStreamGraph());
        String plan = TableJobHelper.streamPlan(streamGraph);
        System.out.println(plan);
        flinkEnv.execute(streamGraph);
    }

    private void trainInputTableStreamEnv(String trainPy) throws Exception {
        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        TableEnvironment tableEnv = TableEnvironment.getTableEnvironment(flinkEnv);
        TFConfig tfConfig = prepareTrainConfig(trainPy);

        WDLTableSource tableSource = new WDLTableSource(tfConfig.getProperty("input") + "/adult.data",
            15, 50000);
        tableEnv.registerTableSource("adult", tableSource);
        Table source = tableEnv.scan("adult");
//        source.writeToSink(new PrintTableSink(TimeZone.getDefault()));

        TFTableJavaUtils.train(flinkEnv, tableEnv, source, tfConfig, null);
        TableJobHelper helper = new TableJobHelper();
        helper.like("WORKER", tfConfig.getWorkerNum());
        helper.like("PS", tfConfig.getPsNum());
        helper.like("AM", 1);
        helper.like("adult", tfConfig.getWorkerNum());
        helper.like("WDLTableSource", tfConfig.getWorkerNum());
        StreamGraph streamGraph =  helper.matchStreamGraph(flinkEnv.getStreamGraph());
        String plan = TableJobHelper.streamPlan(streamGraph);
        System.out.println(plan);
        flinkEnv.execute(streamGraph);
    }



}
