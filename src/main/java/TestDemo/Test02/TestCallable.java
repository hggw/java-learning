package TestDemo.Test02;

import org.apache.catalina.Executor;

import java.util.concurrent.*;

public class TestCallable implements Callable<Boolean> {

    public static String winner;
    @Override
    public Boolean call() throws Exception {
        for (int i = 0; i <= 100; i++) {
            if(gameOver(i)) {
                return false;
            }
            System.out.println(Thread.currentThread().getName()+"跑了" + i + "步");
        }
        return true;
    }
    private boolean gameOver(int steps) {

        if(winner != null) {
            return true;
        }
        if(steps >= 100) {
            winner = Thread.currentThread().getName();
            System.out.println("winner is" + winner);
            return true;
        }
        return false;
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException {

        TestCallable t1 = new TestCallable();
        TestCallable t2 = new TestCallable();
        TestCallable t3 = new TestCallable();

        ExecutorService ser = Executors.newFixedThreadPool(3);

        Future<Boolean> r1 = ser.submit(t1);
        Future<Boolean> r2 = ser.submit(t2);
        Future<Boolean> r3 = ser.submit(t3);

        boolean res1 = r1.get();
        boolean res2 = r2.get();
        boolean res3 = r3.get();
        ser.shutdown();
    }
}
