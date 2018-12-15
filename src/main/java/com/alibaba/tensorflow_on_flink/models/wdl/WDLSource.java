package com.alibaba.tensorflow_on_flink.models.wdl;

import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.ResultTypeQueryable;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.streaming.api.functions.source.ParallelSourceFunction;
import org.apache.flink.types.Row;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * models.
 * 12/15/18.
 */
public class WDLSource implements ResultTypeQueryable, ParallelSourceFunction<Row> {
    private String filePath;
    private int columnNum = 0;
    public  RowTypeInfo typeInfo;
    private long maxNum = Long.MAX_VALUE;
    private long currentNum = 0;

    public WDLSource(String filePath, int column, long maxNum) {
        this.filePath = filePath;
        this.columnNum = column;
        TypeInformation[] types = new TypeInformation[columnNum];
        String[] names = new String[columnNum];
        for(int i = 0; i < columnNum; i++) {
            types[i] = BasicTypeInfo.STRING_TYPE_INFO;
            names[i] = "adult_" + i;
        }
        typeInfo = new RowTypeInfo(types, names);
        this.maxNum = maxNum;
    }

    @Override
    public TypeInformation getProducedType() {
        return typeInfo;
    }

    @Override
    public void run(SourceContext<Row> sourceContext) throws Exception {
        Configuration configuration = new Configuration();
        FileSystem fs = FileSystem.get(configuration);
        FSDataInputStream fsr;
        BufferedReader bufferedReader;
        String lineTxt;
        fsr = fs.open(new Path(filePath));
        bufferedReader = new BufferedReader(new InputStreamReader(fsr));
        List<String> stringList = new ArrayList<>();
        while ((lineTxt = bufferedReader.readLine()) != null)
        {
            stringList.add(lineTxt);
        }
        System.out.println("begin send message");
        boolean flag = true;
        while (flag){
            for(String line: stringList) {
                String[] tmp = line.split(",");
                Row row = new Row(columnNum);
                for(int i = 0; i < columnNum; i++){
                    row.setField(i, tmp[i]);
                }
                sourceContext.collect(row);
                currentNum++;
                if(currentNum > maxNum){
                    flag = false;
                    break;
                }
            }
        }
    }

    @Override
    public void cancel() {

    }
}
