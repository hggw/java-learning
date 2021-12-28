package client;

import com.alibaba.dubbo.config.ApplicationConfig;
import com.alibaba.dubbo.config.ReferenceConfig;
import server.UserService;

import java.lang.ref.Reference;
import java.util.Arrays;

/**
 * @author HGW
 */
public class YoungClient {

    public UserService buildRemoteClient(String remoteUrl) {
        ReferenceConfig<UserService> referenceConfig = new ReferenceConfig<>();
        referenceConfig.setInterface(UserService.class);
        referenceConfig.setUrl(remoteUrl);
        referenceConfig.setApplication(new ApplicationConfig("young-app"));
        return referenceConfig.get();
    }

    public static void main(String[] args) {
        YoungClient client = new YoungClient();
        UserService service = client.buildRemoteClient("dubbo://127.0.0.1:20880/server.UserService");
        System.out.println(service.getUser(111));
    }
}
