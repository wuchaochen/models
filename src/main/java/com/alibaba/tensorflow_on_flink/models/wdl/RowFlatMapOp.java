package com.alibaba.tensorflow_on_flink.models.wdl;

import com.alibaba.flink.tensorflow.client.ExecutionMode;
import com.alibaba.flink.tensorflow.client.TFConfig;
import com.alibaba.flink.tensorflow.util.Role;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.ResultTypeQueryable;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;

public class RowFlatMapOp extends RichFlatMapFunction<Row, Row>
    implements ResultTypeQueryable<Row> {

  private RowMapFunction map;

  public RowFlatMapOp(ExecutionMode mode, Role job, TFConfig config, RowTypeInfo inTI,
      RowTypeInfo outTI) {
    map = new RowMapFunction(mode, job, config, inTI, outTI);
  }

  @Override
  public void open(Configuration parameters) throws Exception {
    map.open(parameters, getRuntimeContext());
  }

  @Override
  public void close() throws Exception {
    map.close();
  }

  @Override
  public void flatMap(Row row, Collector<Row> collector) throws Exception {
    map.flatMap(row, collector);
  }

  @Override
  public TypeInformation<Row> getProducedType() {
    return map.getProducedType();
  }
}
