package MachineLearning.Pipelines;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.SparkSession;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class ExampleMain {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("Dataframe example")
                .master("local[*]")
                .getOrCreate();
        //dati di training
        List<Row> dataTraining = Arrays.asList(
                RowFactory.create(1.0, Vectors.dense(0.0, 1.1, 0.1)),
                RowFactory.create(0.0, Vectors.dense(2.0, 1.0, -1.0)),
                RowFactory.create(0.0, Vectors.dense(2.0, 1.3, 1.0)),
                RowFactory.create(1.0, Vectors.dense(0.0, 1.2, -0.5))
        );
        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });
        Dataset<Row> training = spark.createDataFrame(dataTraining, schema);

        // Istanziazione di regressione logistica (un Estimator)
        LogisticRegression lr = new LogisticRegression();

        System.out.println("LogisticRegression parameters:\n" + lr.explainParams() + "\n");

        //settaggio parametri
        lr.setMaxIter(10).setRegParam(0.01);

        //apprendimento di un modello
        LogisticRegressionModel model1 = lr.fit(training);
        //questo stampa coppie (nome: valore) dove i nomi sono univoci per questa istanza di regressione logistica
        System.out.println("Il primo modello è stato appreso utilizzando i seguenti parametri: "+
                model1.parent().extractParamMap());

        //alternativamente possiamo specificare i parametri utilissando una ParamMap
        ParamMap paramMap = new ParamMap()
                .put(lr.maxIter().w(20))  // Specify 1 Param.
                .put(lr.maxIter(), 30)  // This overwrites the original maxIter.
                .put(lr.regParam().w(0.1), lr.threshold().w(0.55));  // Specify multiple Params.

        // One can also combine ParamMaps.
        ParamMap paramMap2 = new ParamMap()
                .put(lr.probabilityCol().w("myProbability"));  // Change output column name
        ParamMap paramMapCombined = paramMap.$plus$plus(paramMap2);


        //apprendiamo un nuovo modello utilizzando i parametri appena combinati
        LogisticRegressionModel model2 = lr.fit(training, paramMapCombined);
        System.out.println("Il secondo modello è stato appreso utilizzando i seguenti parametri: " +
                model2.parent().extractParamMap());

        //prepariamo i documenti di test
        List<Row> dataTest = Arrays.asList(
                RowFactory.create(1.0, Vectors.dense(-1.0, 1.5, 1.3)),
                RowFactory.create(0.0, Vectors.dense(3.0, 2.0, -0.1)),
                RowFactory.create(1.0, Vectors.dense(0.0, 2.2, -1.5))
        );
        //da notare che lo schema deve essere lo stesso
        Dataset<Row> test = spark.createDataFrame(dataTest, schema);


        System.out.println("---------------------------testing---------------------------");
        Dataset<Row> results = model1.transform(test);
        Dataset<Row> rows = results.select("features", "label", "probability", "prediction");
        for (Row r: rows.collectAsList()) {
            System.out.println("(" + r.get(0) + ", " + r.get(1) + ") -> prob=" + r.get(2) + ", prediction=" + r.get(3));
        }

    }
}
