package TestDemo.Test02;

import java.util.concurrent.*;

public class TestPool {

    public static void main(String[] args) {

        ExecutorService service = Executors.newFixedThreadPool(10);
        service.execute(new MyThread());
        service.execute(new MyThread());
        service.execute(new MyThread());
        service.execute(new MyThread());
        service.shutdown();


    }
}

class MyThread implements Runnable {

    @Override
    public void run() {

        System.out.println(Thread.currentThread().getName());

    }
}