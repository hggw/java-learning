package TestDemo.SpringListenerDemo;

import TestDemo.SpringListenerDemo.MyTestEventPubListener;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

/**
 * @author HGW
 */
@Controller
public class TestEventListenerController {

    @Autowired
    private MyTestEventPubListener publisher;

    @RequestMapping(value = "/test/testPublishEvent1" )
    public void testPublishEvent(){
        publisher.pushListener("我来了！");
    }
}
