package machineLearning.preprcessing.featuresExtractors;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.FeatureHasher;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/*
Feature hashing projects a set of categorical or numerical features into a feature vector of specified dimension
(typically substantially smaller than that of the original feature space).
This is done using the hashing trick to map features to indices in the feature vector.
example:

INPUT table:
real| bool|stringNum|string
----|-----|---------|------
 2.2| true|        1|   foo
 3.3|false|        2|   bar
 4.4|false|        3|   baz
 5.5|false|        4|   foo

OUTPUT table:
real|bool |stringNum|string|features
----|-----|---------|------|-------------------------------------------------------
2.2 |true |1        |foo   |(262144,[51871, 63643,174475,253195],[1.0,1.0,2.2,1.0])
3.3 |false|2        |bar   |(262144,[6031,  80619,140467,174475],[1.0,1.0,1.0,3.3])
4.4 |false|3        |baz   |(262144,[24279,140467,174475,196810],[1.0,1.0,4.4,1.0])
5.5 |false|4        |foo   |(262144,[63643,140467,168512,174475],[1.0,1.0,1.0,5.5])
 */
public class Features2Vec {

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("Java structred Network Word Count")
                .master("local[*]")
                .getOrCreate();

        List<Row> data = Arrays.asList(
                RowFactory.create(2.2, true, "1", "foo"),
                RowFactory.create(3.3, false, "2", "bar"),
                RowFactory.create(4.4, false, "3", "baz"),
                RowFactory.create(5.5, false, "4", "foo")
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("real", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("bool", DataTypes.BooleanType, false, Metadata.empty()),
                new StructField("stringNum", DataTypes.StringType, false, Metadata.empty()),
                new StructField("string", DataTypes.StringType, false, Metadata.empty())
        });

        Dataset<Row> dataset = spark.createDataFrame(data, schema);

        FeatureHasher hasher = new FeatureHasher()
                .setInputCols(new String[]{"real", "bool", "stringNum", "string"})
                .setOutputCol("features");

        Dataset<Row> featurized = hasher.transform(dataset);

        featurized.show();

    }
}
