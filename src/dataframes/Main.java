package dataframes;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.log4j.Logger;
import org.apache.log4j.Level;

public class Main {



    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("Dataframe example")
                .master("local[*]")
                .getOrCreate();

        Dataset<Row> df = spark.read().json("people.json");

        System.out.println("<---------- some query on json dataset ---------->");

        df.show();

        df.printSchema();

        System.out.println("<---------- All names ---------->");
        df.select("name").show();

        System.out.println("<---------- name and ages plus 1 ---------->");
        df.select(df.col("name"), df.col("age").plus(1)).show();

        System.out.println("<---------- ages greather than 21 ---------->");
        df.select(df.col("age").gt(21)).show();

        System.out.println("<---------- group by age and count ---------->");
        df.groupBy("age").count().show();

        System.out.println("<---------- create a temporary view ---------->");

        df.createOrReplaceTempView("people");

        Dataset<Row> sqlDF = spark.sql("SELECT * FROM people");

        sqlDF.show();

        spark.stop();
    }
}
