package TestDemo.SpringListenerDemo;

import TestDemo.SpringListenerDemo.MyTestEvent;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

/**
 * @author HGW
 */
@Component
public class MyTestEventPubListener {
    @Autowired
    private ApplicationContext applicationContext;

    // 事件发布方法
    public void pushListener(String msg) {
        applicationContext.publishEvent(new MyTestEvent(this,msg));
        System.out.println("MyTestEventPubListener going");
    }

}
