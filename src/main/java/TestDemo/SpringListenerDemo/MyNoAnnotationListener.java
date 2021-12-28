package TestDemo.SpringListenerDemo;


import org.springframework.context.ApplicationListener;
import org.springframework.stereotype.Component;


/**
 * @author HGW
 */
@Component
public class MyNoAnnotationListener implements ApplicationListener<MyTestEvent>{

    @Override
    public void onApplicationEvent(MyTestEvent event) {
        if(event instanceof MyTestEvent){
            System.out.println("my event laungh");
        }else{
            System.out.println("other envet");
        }
    }

}
