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

        SparkConf spark = new SparkConf().setAppName("Dataframe example")
                .setMaster("local[*]");

        JavaSparkContext sc = new JavaSparkContext(spark);


        JavaRDD<String> lines = sc.textFile("resources/data.txt");
        JavaRDD<Integer> lineLengths = lines.map(s -> {
            Integer total = 0;
            for(String number : s.split(",")) {
                try {
                    total += Integer.parseInt(number);
                    System.out.print(Integer.parseInt(number)+ " ");
                }catch(NumberFormatException e){
                    System.err.println("\nWARN: impossible parse this character: \""+number+ "\"");
                }
            }
            System.out.println("Total line: "+total);
            return total;
        });
        lineLengths.persist(StorageLevel.MEMORY_ONLY());
        int totalLength = lineLengths.reduce((a, b) -> a + b);

        /*
        //creating an RDD of persons from a text file

        JavaRDD<Person> peopleRDD = spark.read()
                .textFile("people.txt")
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

        Dataset<Row> teenagersDF = spark.sql("SELECT name FROM people WHERE age BETWEEN 13 AND 19");

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
        */
    }
}
