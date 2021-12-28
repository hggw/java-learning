package server.SimpleServer;

import com.alibaba.dubbo.common.URL;
import com.alibaba.dubbo.config.ApplicationConfig;
import com.alibaba.dubbo.config.ProtocolConfig;
import com.alibaba.dubbo.config.RegistryConfig;
import com.alibaba.dubbo.config.ServiceConfig;
import server.UserService;

import javax.xml.ws.Service;
import java.io.IOException;
import java.util.List;

/**
 * @author HGW
 */
public class SimpleServer {

    public void openService(int port) {
        ServiceConfig serviceConfig = new ServiceConfig();
        serviceConfig.setInterface(UserService.class);
        serviceConfig.setProtocol(new ProtocolConfig("dubbo",20880));
        serviceConfig.setRegistry(new RegistryConfig(RegistryConfig.NO_AVAILABLE));
        serviceConfig.setApplication(new ApplicationConfig("simple-app"));
        UserServiceImp ref = new UserServiceImp();
        serviceConfig.setRef(ref);
        serviceConfig.export();
        List<URL> list = serviceConfig.getExportedUrls();
        ref.setPort(list.get(0).getPort());
        System.out.println("服务已开启：" + port);
    }

    public static void main(String[] args) throws IOException {
        new SimpleServer().openService(-1);
        System.in.read();
    }
}
