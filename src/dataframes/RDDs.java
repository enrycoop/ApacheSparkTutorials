package dataframes;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.*;
import org.apache.spark.storage.StorageLevel;

import java.util.Arrays;
import java.util.List;

/*
 * Questo metodo di conversione si può impiegare soltanto
 * se si conosce lo schema dei dati.
 */
public class RDDs {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkConf sparkConf = new SparkConf().setAppName("Dataframe example")
                .setMaster("local[*]");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        JavaRDD<String> lines = sc.textFile("resources/data.txt");
        JavaRDD<Integer> lineLengths = lines.map(s -> {
            Integer total = 0;
            StringBuilder log = new StringBuilder();
            for(String number : s.split(",")) {
                try {
                    total += Integer.parseInt(number);
                    log.append(Integer.parseInt(number)).append(" ");
                }catch(NumberFormatException e){
                    ControllerLogger.warn("impossible parse this character: \""+number+ "\"");
                }catch(Exception e1){
                    ControllerLogger.error("Unexpected Exception "+e1.getMessage());
                }
            }
            ControllerLogger.info(log.append("Total line: "+total).toString());
            return total;
        });
        lineLengths.persist(StorageLevel.MEMORY_ONLY());
        int totalLength = lineLengths.reduce((a, b) -> a + b);
        ControllerLogger.info("total Length: "+totalLength);

        //creating an RDD of persons from a text file
        SparkSession spark = SparkSession.builder()
                .appName("Dataframe example")
                .master("local[*]")
                .getOrCreate();

        JavaRDD<Person> peopleRDD = spark.read()
                .textFile("resources/people.txt")
                .javaRDD()
                .map( line -> {
                    String[] parts = line.split(",");
                    Person person = new Person();
                    person.setName(parts[0]);
                    person.setAge(Integer.parseInt(parts[1]));
                    return person;
                        }
                );
        //convert the RDD in a dataframe
        Dataset<Row> peopleDF = spark.createDataFrame(peopleRDD, Person.class);

        peopleDF.createOrReplaceTempView("people");

        // SQL statements ...
        String query = "SELECT name FROM people WHERE age BETWEEN 13 AND 19";

        Dataset<Row> teenagersDF = spark.sql(query);

        //to access to the column of row you need to a field index
        Encoder<String> stringEncoder = Encoders.STRING();
        Dataset<String> teenagerNames = teenagersDF.map(
                (MapFunction<Row, String>) row -> "Name: " + row.getString(0),
                stringEncoder
        );
        teenagerNames.show();

        // access by field name
        Dataset<String> teenagersByFieldName = teenagersDF.map(
                (MapFunction<Row, String>) row -> "Name: " + row.<String>getAs("name"),
                stringEncoder
        );
        teenagersByFieldName.show();

    }
}
