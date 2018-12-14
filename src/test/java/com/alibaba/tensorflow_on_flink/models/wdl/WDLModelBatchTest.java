package com.alibaba.tensorflow_on_flink.models.wdl;

import com.alibaba.flink.tensorflow.python.ProcessPythonRunner;
import com.alibaba.flink.tensorflow.util.Docker;
import com.alibaba.flink.tensorflow.util.ShellExec;
import com.alibaba.flink.tensorflow.util.SysUtil;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;

import static com.alibaba.tensorflow_on_flink.models.wdl.ClusterUtil.JAR_NAME;

/**
 * models.
 * 12/14/18.
 */
public class WDLModelBatchTest {
    static String hdfsDataDir = "/user/root/data/";


    @Before
    public void setUp() throws Exception {
        ClusterUtil.downloadVenv();
        YarnCluster.start();
        ClusterUtil.copyJarToContainer();
        ClusterUtil.copyVenvToContainer();
        ClusterUtil.copyDataToContainer();
        YarnCluster.prepareHDFSEnv();
    }

    @After
    public void tearDown() throws Exception {
        YarnCluster.stop();
    }

    @Test
    public void testRunWDLBatch() throws Exception{
        System.out.println(ClusterUtil.getProjectRootPath());
    }

    private void runAndVerify(WDLModelBatch.EnvMode mode) {
        String output = YarnCluster.flinkStreamRun(JAR_NAME,
            WDLModelBatch.class.getCanonicalName(),
            "--train-dir",
            hdfsDataDir,
            "--code",
            hdfsDataDir + "code.zip",
            "--output-dir",
            "/user/root/minist/output_" + System.currentTimeMillis(),
            "--zk-conn-str",
            YarnCluster.getZKContainer(),
            "--mode",
            mode.toString(),
            "--train",
            "/mnist_dist.py",
            "--envpath",
            YarnCluster.getVenvHdfsPath(),
            "--runner-class",
            ProcessPythonRunner.class.getCanonicalName()
        );
        System.out.println(output);
        String appId = YarnCluster.parseApplicationId(output);
        String rootPath = ClusterUtil.getProjectRootPath();
        if(!appId.isEmpty()) {
            YarnCluster.dumpFlinkLogs(appId, rootPath,"/usr/local/hadoop/");
        }else{
            System.err.println("appid is empty");
        }
    }
    @Test
    public void testFlinkStreamRun() throws Exception{
        System.out.println(SysUtil._FUNC_());
        String rootPath = ClusterUtil.getProjectRootPath();
        File code = new File(rootPath + "/target/code/");
        if(code.exists()){
            code.delete();
        }
        code.mkdir();
        ShellExec.run("cp -r " + rootPath + "/python/* " + code.getAbsolutePath());
        ShellExec.run("cd " + rootPath + "/target && zip -r  " + rootPath + "/target/code.zip code");
        Docker.copyToContainer(YarnCluster.getYarnContainer(),
            rootPath + "/target/code.zip", YarnCluster.WORK_HOME);
        Docker.exec(YarnCluster.getYarnContainer(), "hadoop fs -put "
            + YarnCluster.WORK_HOME + "/code.zip " + hdfsDataDir);

        Docker.copyToContainer(YarnCluster.getYarnContainer(),
            rootPath + "/examples/mnist/src/python/mnist_dist.py", YarnCluster.WORK_HOME);
        Docker.copyToContainer(YarnCluster.getYarnContainer(),
            rootPath + "/alibaba/target/" + JAR_NAME, YarnCluster.WORK_HOME);
        runAndVerify(WDLModelBatch.EnvMode.StreamEnv);
    }

}