package machineLearning.supervised.classification;

/*
Naive Bayes classifiers are a family of simple probabilistic, multiclass classifiers based on applying Bayes’ theorem
with strong (naive) independence assumptions between every pair of features.
Naive Bayes can be trained very efficiently. With a single pass over the training data, it computes the conditional
probability distribution of each feature given each label. For prediction, it applies Bayes’ theorem to compute
the conditional probability distribution of each label given an observation.

MLlib supports both multinomial naive Bayes and Bernoulli naive Bayes.

Input data: These models are typically used for document classification. Within that context, each observation is
a document and each feature represents a term. A feature’s value is the frequency of the term
(in multinomial Naive Bayes) or a zero or one indicating whether the term was found in the document
(in Bernoulli Naive Bayes). Feature values must be non-negative. The model type is selected with an optional parameter
“multinomial” or “bernoulli” with “multinomial” as the default. For document classification, the input feature vectors
should usually be sparse vectors. Since the training data is only used once, it is not necessary to cache it.

Additive smoothing can be used by setting the parameter λ (default to 1.0).
 */

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class NaiveBayesMain {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);
        SparkSession spark = SparkSession.builder()
                .appName("Java structred Network Word Count")
                .master("local[*]")
                .getOrCreate();

        Dataset<Row> dataFrame =
                spark.read().format("libsvm").load("resources/sample_libsvm_data.txt");

        Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 1234L);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        // create the trainer and set its parameters
        NaiveBayes nb = new NaiveBayes();

        // train the model
        NaiveBayesModel model = nb.fit(train);

        // Select example rows to display.
        Dataset<Row> predictions = model.transform(test);
        predictions.show();

        // compute accuracy on the test set
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test set accuracy = " + accuracy);
    }
}
