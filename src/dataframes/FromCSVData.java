package dataframes;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class FromCSVData {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("Dataframe example")
                .master("local[*]")
                .getOrCreate();


        spark.udf().register("myAverage", new MyAverage());

        Dataset<Row> peopleDF = spark.read().format("csv")
                .option("sep",";")
                .option("inferSchema","true")
                .option("header","true")
                .load("people.csv");
        peopleDF.createOrReplaceTempView("people");
        peopleDF.show();

        Dataset<Row> result = spark.sql("SELECT myAverage(age) FROM people");
        result.show();
    }
}
