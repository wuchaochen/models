package com.alibaba.tensorflow_on_flink.models.wdl;

import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.sinks.PrintTableSink;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.TimeZone;

import static org.junit.Assert.*;

/**
 * models.
 * 12/15/18.
 */
public class WDLTableSourceTest {
    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }
    @Test
    public void testRunWDLSource() throws Exception{
        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        flinkEnv.setParallelism(1);
        TableEnvironment tableEnv = TableEnvironment.getTableEnvironment(flinkEnv);
        String rootPath = ClusterUtil.getProjectRootPath();

        tableEnv.registerTableSource("adult",
            new WDLTableSource("file:///" + rootPath +"/python/data/adult.test", 15, 5));
        Table source = tableEnv.scan("adult");
        source.writeToSink(new PrintTableSink(TimeZone.getDefault()));
        flinkEnv.execute();
    }

}