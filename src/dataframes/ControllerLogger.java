package dataframes;

public class ControllerLogger {
    public static void warn(String log){
        System.out.println("WARN| "+log);
    }

    public static void info(String log){
        System.out.println("INFO| "+log);
    }

    public static void error(String log){
        System.err.println("ERROR| "+log);
    }
}
