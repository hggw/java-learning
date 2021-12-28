package TestDemo.Test02;

public interface Marry {

    void happyMarry();
}

class You implements Marry{

    @Override
    public void happyMarry() {
        System.out.println("黄老师要结婚了，好开心!");
    }
}

class Wedding implements Marry {

    private Marry target;

    public Wedding(Marry target) {
        this.target = target;
    }

    @Override
    public void happyMarry() {
        before();
        this.target.happyMarry();
        after();
    }

    private void after() {

        System.out.println("收尾款");
    }

    private void before() {
        System.out.println("布置现场");
    }

    public static void main(String[] args) {
        Wedding test = new Wedding(new You());
        test.happyMarry();
    }
}
