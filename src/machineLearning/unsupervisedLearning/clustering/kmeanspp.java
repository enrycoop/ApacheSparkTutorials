package machineLearning.unsupervisedLearning.clustering;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.BisectingKMeans;
import org.apache.spark.ml.clustering.BisectingKMeansModel;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class kmeanspp {

    private static void kmeans(SparkSession spark){
        // Loads data.
        Dataset<Row> dataset = spark.read().format("libsvm").load("resources/sample_kmeans_data.txt");

        // Trains a k-means model.
        KMeans kmeans = new KMeans().setK(2).setSeed(1L);
        KMeansModel model = kmeans.fit(dataset);

        // Make predictions
        Dataset<Row> predictions = model.transform(dataset);

        // Evaluate clustering by computing Silhouette score
        ClusteringEvaluator evaluator = new ClusteringEvaluator();

        double silhouette = evaluator.evaluate(predictions);
        System.out.println("Silhouette with squared euclidean distance = " + silhouette);

        // Shows the result.
        Vector[] centers = model.clusterCenters();
        System.out.println("Cluster Centers: ");
        for (Vector center: centers) {
            System.out.println(center);
        }
    }

    private static void bisectingkmeans(SparkSession spark){
        // Loads data.
        Dataset<Row> dataset = spark.read().format("libsvm").load("resources/sample_kmeans_data.txt");
        // Trains a bisecting k-means model.
        BisectingKMeans bkm = new BisectingKMeans().setK(2).setSeed(1);
        BisectingKMeansModel model = bkm.fit(dataset);

        // Evaluate clustering.
        double cost = model.computeCost(dataset);
        System.out.println("Within Set Sum of Squared Errors = " + cost);

        // Shows the result.
        System.out.println("Cluster Centers: ");
        Vector[] centers = model.clusterCenters();
        for (Vector center : centers) {
            System.out.println(center);
        }
    }

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("App")
                .master("local[*]")
                .getOrCreate();

        kmeanspp.bisectingkmeans(spark);

    }
}
