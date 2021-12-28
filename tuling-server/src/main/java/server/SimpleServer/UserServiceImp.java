package server.SimpleServer;

import server.UserService;

/**
 * @author HGW
 */
public class UserServiceImp implements UserService {
    @Override
    public String getUser(Integer id) {
        return "返回id:"+id;
    }

    public void setPort(int port) {
        System.out.println("返回port = " + port);
    }
}
