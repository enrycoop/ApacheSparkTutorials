package machineLearning.unsupervisedLearning.clustering;
/*
A Gaussian Mixture Model represents a composite distribution whereby points are drawn from
 one of k Gaussian sub-distributions, each with its own probability. The spark.ml implementation
 uses the expectation-maximization algorithm to induce the maximum-likelihood model given a set of samples.
 Input Columns
featuresCol	Vector	"features"
Output Columns
predictionCol	Int	"prediction"	Predicted cluster center
probabilityCol	Vector	"probability"	Probability of each cluster
 */
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.GaussianMixture;
import org.apache.spark.ml.clustering.GaussianMixtureModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GMM {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("App")
                .master("local[*]")
                .getOrCreate();
        // Loads data
        Dataset<Row> dataset = spark.read().format("libsvm").load("resources/sample_kmeans_data.txt");

        // Trains a GaussianMixture model
        GaussianMixture gmm = new GaussianMixture()
                .setK(2);
        GaussianMixtureModel model = gmm.fit(dataset);

        // Output the parameters of the mixture model
        for (int i = 0; i < model.getK(); i++) {
            System.out.printf("Gaussian %d:\nweight=%f\nmu=%s\nsigma=\n%s\n\n",
                    i, model.weights()[i], model.gaussians()[i].mean(), model.gaussians()[i].cov());
        }
    }
}
