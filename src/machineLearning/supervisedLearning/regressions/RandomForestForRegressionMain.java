package machineLearning.supervisedLearning.regressions;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;

public class RandomForestForRegressionMain {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("App")
                .master("local[*]")
                .getOrCreate();

        // Load training data.
        Dataset<Row> data = spark.read().format("libsvm")
                .load("resources/sample_libsvm_data.txt");

        // Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4)
                .fit(data);

// Split the data into training and test sets (30% held out for testing).
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

// Train a DecisionTree model.
        RandomForestRegressor dt = new RandomForestRegressor()
                .setLabelCol("label")
                .setFeaturesCol("indexedFeatures");

// Chain indexer and tree in a Pipeline.
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{featureIndexer, dt});

// Train model. This also runs the indexer.
        PipelineModel model = pipeline.fit(trainingData);

// Make predictions.
        Dataset<Row> predictions = model.transform(testData);

// Select example rows to display.
        predictions.select("prediction","label", "features").show(5);

// Select (prediction, true label) and compute test error.
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        double rmse = evaluator.evaluate(predictions);
        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

        RandomForestRegressionModel treeModel =
                (RandomForestRegressionModel) (model.stages()[1]);
        System.out.println("Learned regression tree model:\n" + treeModel.toDebugString());
    }
}
