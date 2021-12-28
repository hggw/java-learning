package TestDemo.Test02;

public class TestRunnable implements Runnable{
    private int count = 10;

    @Override
    public void run() {
        while(true) {
            if(count <= 0) {
                break;
            }
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.println(Thread.currentThread().getName()+"抢到第"+count--+"张票");
        }
    }

    public static void main(String[] args) {
        TestRunnable t = new TestRunnable();

        new Thread(t,"小黄").start();
        new Thread(t,"小杨").start();
        new Thread(t,"小黄牛").start();
    }
}
