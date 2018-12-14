package com.alibaba.tensorflow_on_flink.models.wdl;

import com.alibaba.flink.tensorflow.util.Docker;
import com.alibaba.flink.tensorflow.util.MiniYarnCluster;
import com.alibaba.flink.tensorflow.util.ShellExec;
import com.alibaba.flink.tensorflow.util.TestUtil;
import com.google.common.base.Preconditions;
import org.apache.maven.model.Model;
import org.apache.maven.model.io.xpp3.MavenXpp3Reader;
import org.codehaus.plexus.util.xml.pull.XmlPullParserException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/**
 * models.
 * 12/14/18.
 */
public class ClusterUtil {
    private static Logger LOG = LoggerFactory.getLogger(ClusterUtil.class);

    private static String rootPath = null;
    private static final String PARENT_NAME = "models";

    public static String JAR_NAME = "models-1.0-SNAPSHOT.jar";

    public static boolean copyJarToContainer(){
        String rootPath = getProjectRootPath();
        return Docker.copyToContainer(MiniYarnCluster.getYarnContainer(),
            rootPath + "/target/" + JAR_NAME,
            MiniYarnCluster.WORK_HOME + "/" + JAR_NAME);
    }

    public static boolean copyVenvToContainer(){
        String rootPath = getProjectRootPath();
        return Docker.copyToContainer(MiniYarnCluster.getYarnContainer(),
            rootPath + "/target/" + MiniYarnCluster.VENV_PACK,
            MiniYarnCluster.VENV_LOCAL_PATH);
    }


    public static boolean copyDataToContainer(){
        String rootPath = getProjectRootPath();
        boolean res = false;
        res = Docker.exec(MiniYarnCluster.getYarnContainer(), "mkdir " + MiniYarnCluster.WORK_HOME + "/data/");
        if(!res){
            return res;
        }
        res = Docker.copyToContainer(MiniYarnCluster.getYarnContainer(),rootPath + "/python/data/.",
            MiniYarnCluster.WORK_HOME + "/data/");
        return res;
    }

    public static String getProjectRootPath() {
        if (rootPath == null) {
            // assume the working dir is under root
            File file = new File(System.getProperty("user.dir"));
            while (file != null) {
                File pom = new File(file, "pom.xml");
                if (pom.exists()) {
                    try (FileReader fileReader = new FileReader(pom)) {
                        MavenXpp3Reader reader = new MavenXpp3Reader();
                        Model model = reader.read(fileReader);
                        if (model.getArtifactId().equals(PARENT_NAME)) {
                            rootPath = file.getAbsolutePath();
                            break;
                        }
                    } catch (XmlPullParserException | IOException e) {
                        LOG.error("Error reading pom files", e);
                        break;
                    }
                }
                file = file.getParentFile();
            }
        }
        Preconditions.checkState(rootPath != null, "Cannot determine the project's root path");
        return rootPath;
    }

    public static void downloadVenv(){
        String rootPath = getProjectRootPath();
        File tfenv = new File(rootPath + "/target/tfenv.zip");
        if(!tfenv.exists()) {
            ShellExec.run("wget http://etaose.oss-cn-hangzhou-zmf.aliyuncs.com/test%2Ftf_on_flink%2Fopen%2Ftfenv.zip -O "
                + rootPath + "/target/tfenv.zip");
        }
    }
}
