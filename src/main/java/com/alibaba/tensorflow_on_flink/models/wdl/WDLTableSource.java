package com.alibaba.tensorflow_on_flink.models.wdl;

import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.table.api.types.DataType;
import org.apache.flink.table.api.types.DataTypes;
import org.apache.flink.table.api.types.InternalType;
import org.apache.flink.table.plan.stats.TableStats;
import org.apache.flink.table.sources.StreamTableSource;
import org.apache.flink.types.Row;

import java.io.Serializable;

/**
 * models.
 * 12/15/18.
 */
public class WDLTableSource implements StreamTableSource<Row>, Serializable {
    private WDLSource source;
    private RowTypeInfo typeInfo;
    public WDLTableSource(String path, int num, long max) {
        this.source = new WDLSource(path, num, max);
        typeInfo = source.typeInfo;
    }

    @Override
    public DataType getReturnType() {
        return DataTypes.of(typeInfo);
    }

    @Override
    public TableSchema getTableSchema() {
        InternalType[] types = new InternalType[typeInfo.getArity()];
        for(int i = 0; i < types.length; i++){
            types[i] = DataTypes.internal(typeInfo.getTypeAt(i));
        }
        return new TableSchema(typeInfo.getFieldNames(), types);
    }

    @Override
    public String explainSource() {
        return "WDLTableSource";
    }

    @Override
    public TableStats getTableStats() {
        return null;
    }

    @Override
    public DataStream<Row> getDataStream(StreamExecutionEnvironment execEnv) {
        return execEnv.addSource(source)
            .name(explainSource());
    }

    public RowTypeInfo getTypeInfo() {
        return typeInfo;
    }

}
