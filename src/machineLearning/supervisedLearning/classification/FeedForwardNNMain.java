package machineLearning.supervisedLearning.classification;

/*
Multilayer perceptron classifier (MLPC) is a classifier based on the feedforward artificial neural network.
MLPC consists of multiple layers of nodes. Each layer is fully connected to the next layer in the network.
Nodes in the input layer represent the input data. All other nodes map inputs to outputs by a linear combination
of the inputs with the node’s weights w and bias b and applying an activation function.
Nodes in intermediate layers use sigmoid (logistic) function.
Nodes in the output layer use softmax function.
The number of nodes N in the output layer corresponds to the number of classes.

MLPC employs backpropagation for learning the model.
We use the logistic loss function for optimization and L-BFGS as an optimization routine.

*/
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class FeedForwardNNMain {

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("Java structred Network Word Count")
                .master("local[*]")
                .getOrCreate();

        // Load and parse the data file, converting it to a DataFrame.
        Dataset<Row> dataFrame = spark.read().format("libsvm").load("resources/multiclass.data");

        // Split the data into train and test
        Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 1234L);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        // specify layers for the neural network:
        // input layer of size 4 (features), two intermediate of size 5 and 4
        // and output of size 3 (classes)
        int[] layers = new int[] {4, 5, 4, 3};

        // create the trainer and set its parameters
        MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(1234L)
                .setMaxIter(100);


        // train the model
        MultilayerPerceptronClassificationModel model = trainer.fit(train);

        // compute accuracy on the test set
        Dataset<Row> result = model.transform(test);
        Dataset<Row> predictionAndLabels = result.select("prediction", "label");
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy");

        System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));



    }

}
