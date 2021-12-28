package TestDemo.Test02;

public class Race implements Runnable{

    public static String winner;
    @Override
    public void run() {
        for (int i = 0; i <= 100; i++) {
            if(gameOver(i)) {
                break;
            }
            System.out.println(Thread.currentThread().getName()+"跑了" + i + "步");
        }
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

    public static void main(String[] args) {
        Race race = new Race();
        new Thread(race,"兔子").start();
        new Thread(race,"乌龟").start();
    }
}
