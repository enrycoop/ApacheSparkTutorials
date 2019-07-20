package machineLearning.preprcessing;


import java.io.*;
import java.util.Scanner;

public class CSV2libsvm {
    /*
    the format is csv with separator in input and
     */
    public static void convert(String sep, String inPath, String outPath) throws IOException {
        Scanner sc = new Scanner(new File(inPath));
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outPath));
        System.out.println("Start processing data...");
        while(sc.hasNext()){
            String[] fields = sc.next().split(sep);
            StringBuilder line = null;

            line = new StringBuilder(fields[fields.length - 1]);


            for(int i = 0; i<fields.length-1;i++){
                line.append(" ").append((i+1)).append(":").append(fields[i]);
            }
            line.append("\n");
            bufferedWriter.write(String.valueOf(line));
        }

        bufferedWriter.flush();
        bufferedWriter.close();
        sc.close();
        System.out.println("Conversion complete.");
    }

    public static void main(String[] args) throws IOException {
        CSV2libsvm.convert(",","resources/iris.data","resources/iris.txt");
    }
}
