package com.alibaba.tensorflow_on_flink.models.wdl;

import com.alibaba.flink.tensorflow.util.Docker;
import com.alibaba.flink.tensorflow.util.ShellExec;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class YarnCluster {

    private static Logger LOG = LoggerFactory.getLogger(YarnCluster.class);
    public static final String WORK_HOME = "/usr/local/";
    public static final String ZK_IMAGE = "zookeeper";
    public static final String ZK_SERVER_NAME = "minizk";
    public static final String HDFS_HOME = "/opt/hadoop-2.7.0";
    public static final int HDFS_PORT = 9000;
    public static final String VENV_PACK = "tfenv.zip";
    public static final String VENV_LOCAL_PATH = WORK_HOME + VENV_PACK;
    public static final String VENV_HDFS_PATH = "/user/root/";
    public static final String YARN_NAME = "hadoop-master";
    public static final String YARN_IMAGE = "yarn:v1";
    private static final String ALI_FLINK_YARN_IMAGE = "yarn:aliflink";
    public static final String YARN_ORG_IMAGE = "prographerj/centos7-hadoop:latest";
    public static final String YARN_CMD = "sh -x /etc/bootstrap.sh -d";
    public static final String FLINK_HOME = "/usr/local/flink";
    public static final String HADOOP_HOME = "/usr/local/hadoop";
    public static final String FLINK_CMD= FLINK_HOME + "/bin/flink";
    public static final String HADOOP_CMD= HADOOP_HOME + "/bin/hadoop";
    public static final String JAR_NAME = ClusterUtil.JAR_NAME;

    private final long id;

    private YarnCluster(long id) {
        this.id = id;
    }

    public static void waitClusterReady(){
        boolean flag = false;
        while (!flag) {
            flag = ShellExec.run("curl http://localhost:8088", true);
            try {
                Thread.sleep(5000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }



    public static boolean prepareHDFSEnv(){
        boolean res = Docker.exec(getYarnContainer(), "hadoop fs -mkdir -p " + VENV_HDFS_PATH);
        if(!res){
            return res;
        }
        res = Docker.exec(getYarnContainer(), "hadoop fs -put " + WORK_HOME + VENV_PACK + " " +VENV_HDFS_PATH);
        if(!res){
            return res;
        }
        return res;
    }

    public static YarnCluster start() {
        // make sure to use the latest image
        Docker.pull(YARN_ORG_IMAGE);
        if(!Docker.imageExist("yarn")){
            String rootPath = ClusterUtil.getProjectRootPath();
            String centosDockerDir = rootPath + "/yarn-docker/centos-7/";
            Preconditions.checkState(ShellExec.run("cd " + centosDockerDir +" && docker build -t yarn:v1 .", LOG::info),
                "Failed to build yarn image");
        }
        if (!Docker.imageExist(ALI_FLINK_YARN_IMAGE)) {
            String rootPath = ClusterUtil.getProjectRootPath();
            String dockerDir = rootPath + "/yarn-docker/aliflink";
            Preconditions.checkState(ShellExec.run(
                    String.format("cd %s && docker build -t %s .", dockerDir, ALI_FLINK_YARN_IMAGE), LOG::info),
                    "Failed to build image: " + ALI_FLINK_YARN_IMAGE);
        }

        YarnCluster cluster = new YarnCluster(System.currentTimeMillis());
        try {
            Preconditions.checkState(YarnCluster.startZookeeper(), "Failed to start Zookeeper");
            Preconditions.checkState(YarnCluster.startYarn(), "Failed to start Yarn cluster");
            waitClusterReady();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return cluster;
    }

    public static void stop() {
        Docker.killAndRemoveContainer(getZKContainer());
        Docker.killAndRemoveContainer(getYarnContainer());
    }

    public static String flinkStreamRun(String jarName, String className, String... args) {
        StringBuffer buffer = new StringBuffer();
        Docker.exec(getYarnContainer(), String.format(FLINK_CMD
                + " run -m yarn-cluster -yst -yn 2 -yjm 2048 -ytm 2048 -yD taskmanager.network.memory.max=268435456 " +
//                + " run -m yarn-cluster -yst -yn 1 -yjm 2048 -ytm 6104 " +
                "-c %s %s %s",
            className, WORK_HOME + jarName, Joiner.on(" ").join(args)), buffer);
        return buffer.toString();
    }

    public static String flinkBatchRun(String className, String... args) {
        StringBuffer buffer = new StringBuffer();
        Docker.exec(getYarnContainer(), String.format(FLINK_CMD
                + " run -m yarn-cluster -yn 4 -yjm 1024 -ytm 1024 " +
                "-c %s %s %s",
            className, uberJar(), Joiner.on(" ").join(args)), buffer);
        return buffer.toString();
    }

    public static void dumpFlinkLogs(String appId, String rootPath, String hadoopHome) {
        File appLogDir = new File(rootPath + "/target/" + appId);
        if(!appLogDir.exists()){
            appLogDir.mkdir();
        }

        final String yarnLogDir = hadoopHome + "/logs/userlogs/" + appId + "/.";
        if (!Docker.copyFromContainer(getYarnContainer(), yarnLogDir, appLogDir.getAbsolutePath() + "/")) {
            LOG.warn("Failed to dump logs for " + getYarnContainer());
        }
    }

    public static String parseApplicationId(String log){
        String identify = "Submitting application master";
        String[] lines = log.split("\n");
        for(String line : lines){
            int index = line.indexOf(identify);
            if(index > 0){
                return line.substring(index+identify.length() + 1, line.length());
            }
        }
        return "";
    }

    private static boolean startZookeeper() {
        Docker.ContainerBuilder builder = new Docker.ContainerBuilder();
        builder.image(ZK_IMAGE).cmd("").name(getZKContainer()).opts("-d");
        return builder.build();
    }


    public static boolean startYarn() {
        LOG.info("Starting Yarn...");
        List<Integer> ports = new ArrayList<>();
        ports.add(8088);
        ports.add(50070);
        ports.add(16666);
        Docker.ContainerBuilder builder = new Docker.ContainerBuilder();
        builder.name(getYarnContainer()).cmd(YARN_CMD).image(ALI_FLINK_YARN_IMAGE);
        builder.linkHosts(getZKContainer());
        builder.opts(Collections.singletonList("-d"));
        builder.mapPorts(ports);
        return builder.build();
    }

    public static boolean yarnFlinkRun(String...args){
        String cmd = Joiner.on(" ").join(args);
        return Docker.exec(getYarnContainer(), FLINK_CMD + " run " + cmd);
    }

    private boolean uploadVirtualEnv() {
        boolean res = Docker.exec(getYarnContainer(), HADOOP_CMD + " fs -mkdir -p " + VENV_HDFS_PATH);
        if(!res){
            return res;
        }
        return Docker.execSilently(getYarnContainer(), HADOOP_CMD + " fs -put -f " + VENV_LOCAL_PATH
            + " " + VENV_HDFS_PATH + VENV_PACK);
    }

    public static String getYarnContainer() {
        return toContainerName(YARN_NAME);
    }

    public static String getZKContainer() {
        return toContainerName(ZK_SERVER_NAME);
    }

    private static String toContainerName(String name) {
        // for now we only support one cluster instance
        return name;
    }

    private static String hdfsConfDir() {
        return "file://" + HDFS_HOME + "/etc/hadoop/";
    }

    private static String uberJar() {
        return WORK_HOME + JAR_NAME;
    }

    public static String getVenvHdfsPath() {
        return String.format("hdfs://%s:%d%s", getYarnContainer(), HDFS_PORT, VENV_HDFS_PATH + VENV_PACK);
    }

    public boolean copyToYarn(String src, String dest) {
        return Docker.copyToContainer(getYarnContainer(), src, dest);
    }

}
