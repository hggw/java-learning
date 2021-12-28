package TestDemo.Test02.Syn;

import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.locks.ReentrantLock;

public class Syn1 {
    public static void main(String[] args) {
        TestLock t = new TestLock();
        new Thread(t).start();
        new Thread(t).start();
        new Thread(t).start();


    }
}

class TestLock implements Runnable{
    int ticketNums = 10;
    private final ReentrantLock lock = new ReentrantLock();

    @Override
    public void run() {

        while (true) {
            try {
                lock.lock();
                if (ticketNums > 0) {
                    Thread.sleep(1000);
                    if (ticketNums > 0) {
                        System.out.println(ticketNums--);
                    } else {
                        break;
                    }
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }finally{
                lock.unlock();
            }
        }
    }
}