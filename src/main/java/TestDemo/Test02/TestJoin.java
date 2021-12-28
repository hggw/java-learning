package TestDemo.Test02;

import java.util.Arrays;

public class TestJoin implements Runnable{
    @Override
    public void run() {
        for (int i = 0; i < 10; i++) {
            System.out.println("VIP来了"+ i);
        }
    }

    public static void main(String[] args) throws InterruptedException {
        TestJoin t = new TestJoin();
        Thread t1 = new Thread(t);
        t1.start();

        for (int i = 0; i < 100; i++) {
            if(i == 60) {
                t1.join();
            }
            System.out.println("Main " + i);
        }
    }
}
