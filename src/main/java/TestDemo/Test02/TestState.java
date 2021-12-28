package TestDemo.Test02;

import sun.awt.windows.ThemeReader;

import java.util.Arrays;
import java.util.concurrent.ThreadPoolExecutor;

public class TestState {

    public static void main(String[] args) throws InterruptedException {
        Thread thread = new Thread(() ->{
            for (int i = 0; i < 5; i++) {
                try{
                    Thread.sleep(1000);
                }catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            System.out.println("///////" );
        });

        Thread.State state = thread.getState();
        System.out.println(state);

        thread.start();
        state = thread.getState();
        thread.setPriority(1);
        System.out.println(state);

        while(state != Thread.State.TERMINATED) {
            Thread.sleep(100);
            state = thread.getState();
            System.out.println(state);
        }
    }
}
