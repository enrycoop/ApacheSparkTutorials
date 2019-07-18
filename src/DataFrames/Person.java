package DataFrames;

import java.util.Arrays;
import java.util.Collections;
import java.io.Serializable;


import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.*;

//bean class People
public class Person implements Serializable{
    private String name;
    private int age;

    public String getName(){
        return name;
    }

    public void setName(String name){
        this.name = name;
    }

    public int getAge(){
        return age;
    }

    public void setAge(int age){
        this.age = age;
    }

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("Dataframe example")
                .master("local[*]")
                .getOrCreate();

        //istanza di una bean class
        Person person = new Person();
        person.setName("Enrico");
        person.setAge(27);

        //sono creati encoder per java beans
        Encoder<Person> personEncoder = Encoders.bean(Person.class);
        Dataset<Person> javaBeanDS = spark.createDataset(
                Collections.singletonList(person),
                personEncoder
        );

        javaBeanDS.show();

        //encoders per i tipi più comuni sono forniti dalla classe Encoders
        Encoder<Integer> integerEncoder = Encoders.INT();
        Dataset<Integer> primitiveDS = spark.createDataset(Arrays.asList(1,2,3,4,5),integerEncoder);
        Dataset<Integer> transformedDS = primitiveDS.map(
                (MapFunction<Integer, Integer>) value -> value + 1,
                integerEncoder
        );
        transformedDS.collect(); //restituisce [1,2,3,4,5]

        String path = "people.json";
        Dataset<Person> peopleDS = spark.read().json(path).as(personEncoder);
        peopleDS.show();




    }
}


