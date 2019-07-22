package machineLearning.supervisedLearning.classification;
/*
Logistic regression is a popular method to predict a categorical response. It is a special case of Generalized Linear models that
predicts the probability of the outcomes. In spark.ml logistic regression can be used to predict a binary outcome by using
binomial logistic regression, or it can be used to predict a multiclass outcome by using multinomial logistic regression.
Use the family parameter to select between these two algorithms, or leave it unset and Spark will infer the correct variant.
 */

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.*;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class IrisMain {
    public static void main(String[] args) throws FileNotFoundException {
        long inizio = System.currentTimeMillis();
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("Java structred Network Word Count")
                .master("local[*]")
                .getOrCreate();

        Scanner sc = new Scanner(new File("resources/iris.data"));
        List<Row> data = new ArrayList<>();
        while(sc.hasNext()){
            String[] fields = sc.next().split(",");
            data.add(RowFactory.create(
                    Vectors.dense(Double.parseDouble(fields[0]),Double.parseDouble(fields[1]),Double.parseDouble(fields[2]),Double.parseDouble(fields[3]))
                    ,fields[4].equals("Iris-setosa")?1.0:0.0)
            );

        }

        sc.close();

        StructType schema = new StructType(new StructField[]{
                new StructField("features", new VectorUDT(),false, Metadata.empty()),
                new StructField("label", DataTypes.DoubleType,false, Metadata.empty())
        });



        Dataset<Row> training = spark.createDataFrame(data,schema);


        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        //fit the model
        LogisticRegressionModel model = lr.fit(training);

        // Print the coefficients and intercept for logistic regression
        System.out.println("Coefficients: "
                + model.coefficients() + " Intercept: " + model.intercept());
        /*
        LogisticRegressionTrainingSummary provides a summary for a LogisticRegressionModel.
        In the case of binary classification, certain additional metrics are available, e.g. ROC curve.
        The binary summary can be accessed via the binarySummary method. See BinaryLogisticRegressionTrainingSummary.
        */
        BinaryLogisticRegressionTrainingSummary trainingSummary = model.binarySummary();

        //obtain the loss per iteration.
        double[] objectiveHistory = trainingSummary.objectiveHistory();
        for (double lossPerIteration : objectiveHistory) {
            System.out.println(lossPerIteration);
        }

        //Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
        Dataset<Row> roc = trainingSummary.roc();
        roc.show();
        roc.select("FPR").show();
        System.out.println(trainingSummary.areaUnderROC());


        // Get the threshold corresponding to the maximum F-Measure and rerun LogisticRegression with
        // this selected threshold.
        Dataset<Row> fMeasure = trainingSummary.fMeasureByThreshold();
        double maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0);
        double bestThreshold = fMeasure.where(fMeasure.col("F-Measure").equalTo(maxFMeasure))
                .select("threshold").head().getDouble(0);
        model.setThreshold(bestThreshold);
        long fine = System.currentTimeMillis();
        long time=(fine-inizio);
        System.out.println("execution time: "+time + " millisec");


    }
}
