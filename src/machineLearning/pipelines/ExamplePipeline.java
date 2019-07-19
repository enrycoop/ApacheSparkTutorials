package machineLearning.pipelines;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/*
Una Pipeline concatena più Transformers(modelli) ed Estimators(algoritmi di apprendimento)
insieme per specificare un workflow di Machine Learning.
 */
public class ExamplePipeline {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("Dataframe example")
                .master("local[*]")
                .getOrCreate();

        // Prepare training documents, which are labeled.

        List<Row> data = Arrays.asList(
                RowFactory.create(0L,"a b c d e spark", 1.0),
                RowFactory.create(1L,"b d", 0.0),
                RowFactory.create(2L,"spark f g h", 1.0),
                RowFactory.create(3L,"hadoop mapreduce", 0.0)
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.LongType, false, Metadata.empty()),
                new StructField("text", DataTypes.StringType, false, Metadata.empty()),
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty())
        });
        Dataset<Row> training = spark.createDataFrame(data, schema);

        // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("text")
                .setOutputCol("words");
        HashingTF hashingTF = new HashingTF()
                .setNumFeatures(1000)
                .setInputCol(tokenizer.getOutputCol())
                .setOutputCol("features");
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.001);
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{tokenizer, hashingTF, lr});

        // Fit the pipeline to training documents.
        PipelineModel model = pipeline.fit(training);

        // Prepare test documents, which are unlabeled.
        List<Row> testData = Arrays.asList(
                RowFactory.create(4L,"spark i j k", 1.0),
                RowFactory.create(5L,"l m n",0.0),
                RowFactory.create(6L,"spark hadoop spark",1.0),
                RowFactory.create(7L,"apache hadoop",0.0)
        );
        Dataset<Row> test = spark.createDataFrame(testData, schema);
        // Make predictions on test documents.
        Dataset<Row> predictions = model.transform(test);
        for (Row r : predictions.select("id", "text", "probability", "prediction").collectAsList()) {
            System.out.println("(" + r.get(0) + ", " + r.get(1) + ") --> prob=" + r.get(2)
                    + ", prediction=" + r.get(3));
        }
    }
}

