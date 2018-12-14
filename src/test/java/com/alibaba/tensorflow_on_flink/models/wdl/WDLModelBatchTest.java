package com.alibaba.tensorflow_on_flink.models.wdl;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * models.
 * 12/14/18.
 */
public class WDLModelBatchTest {


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

}