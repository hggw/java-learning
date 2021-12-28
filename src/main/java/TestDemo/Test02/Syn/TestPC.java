package TestDemo.Test02.Syn;

public class TestPC {
    public static void main(String[] args) {
        SyncContainer container = new SyncContainer();
        new Productor(container).start();
        new Consumer(container).start();
    }
}

class Productor extends Thread {
    SyncContainer container;

    public Productor(SyncContainer syncContainer) {
        this.container = syncContainer;
    }
    
    @Override
    public void run() {
        for (int i = 0; i < 100; i++) {
            container.push(new Chicken(i));
            System.out.println("生产了"+i+"只鸡");
        }
    }


}

class Consumer extends Thread {

    SyncContainer container;

    public Consumer(SyncContainer syncContainer) {
        this.container = syncContainer;
    }
    @Override
    public void run() {
        for (int i = 0; i < 100; i++) {
            System.out.println("消费了"+container.pop().id +"只鸡");
        }
    }
}

class Chicken {
    int id;

    public Chicken(int id) {
        this.id = id;
    }
}

class SyncContainer {

    Chicken[] chickens = new Chicken[10];
    int count = 0;

    public synchronized void push(Chicken chicken) {
        if(count == chickens.length) {
            try {
                this.wait();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        chickens[count] = chicken;
        count++;
        this.notifyAll();
    }

    public synchronized Chicken pop() {
        if(count == 0) {
            try {
                this.wait();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        count--;
        Chicken chicken = chickens[count];
        this.notifyAll();
        return chicken;

    }
}