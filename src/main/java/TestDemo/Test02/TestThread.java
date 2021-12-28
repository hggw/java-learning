package TestDemo.Test02;

import java.util.concurrent.ThreadPoolExecutor;

public class TestThread extends Thread{

    @Override
    public void run(){
        for(int i = 0; i < 10; i++) {
            System.out.println("A第"+i+"次");
        }
    }
    public static void main(String[] args) {
        TestThread t = new TestThread();
        t.setDaemon(true);
        t.start();
        // run是先执行完，再往下执行
        t.run();
        for(int i = 0; i<10000; i++) {
            System.out.println("BW第"+i+"次");
        }
        System.out.println(t.getState());
    }
}
