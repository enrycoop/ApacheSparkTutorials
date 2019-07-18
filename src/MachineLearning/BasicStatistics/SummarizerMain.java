package MachineLearning.BasicStatistics;
import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.stat.Summarizer;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;



public class SummarizerMain {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("Summarizer")
                .master("local[*]")
                .getOrCreate();

        //data construction
        List<Row> data = Arrays.asList(
                RowFactory.create(Vectors.dense(2.0, 3.0, 5.0), 1.0),
                RowFactory.create(Vectors.dense(4.0, 6.0, 7.0), 2.0)
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("features", new VectorUDT(), false, Metadata.empty()),
                new StructField("weight", DataTypes.DoubleType, false, Metadata.empty())
        });

        Dataset<Row> df = spark.createDataFrame(data, schema);

        Row result1 = df.select(Summarizer.metrics("mean", "variance")
                .summary(new Column("features"), new Column("weight")).as("summary"))
                .select("summary.mean", "summary.variance").first();
        System.out.println("with weight: mean = " + result1.<Vector>getAs(0).toString() +
                ", variance = " + result1.<Vector>getAs(1).toString());


        Row result2 = df.select(
                Summarizer.mean(new Column("features")),
                Summarizer.variance(new Column("features"))
        ).first();
        System.out.println("without weight: mean = " + result2.<Vector>getAs(0).toString() +
                ", variance = " + result2.<Vector>getAs(1).toString());

    }

}
