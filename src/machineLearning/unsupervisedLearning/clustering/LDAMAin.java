package machineLearning.unsupervisedLearning.clustering;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class LDAMAin {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("App")
                .master("local[*]")
                .getOrCreate();
        // Loads data.
        Dataset<Row> dataset = spark.read().format("libsvm")
                .load("resources/sample_lda_libsvm_data.txt");

        // Trains a LDA model.
        LDA lda = new LDA().setK(10).setMaxIter(10);
        LDAModel model = lda.fit(dataset);

        double ll = model.logLikelihood(dataset);
        double lp = model.logPerplexity(dataset);
        System.out.println("The lower bound on the log likelihood of the entire corpus: " + ll);
        System.out.println("The upper bound on perplexity: " + lp);

        // Describe topics.
        Dataset<Row> topics = model.describeTopics(3);
        System.out.println("The topics described by their top-weighted terms:");
        topics.show(false);

        // Shows the result.
        Dataset<Row> transformed = model.transform(dataset);
        transformed.show(false);

    }
}
