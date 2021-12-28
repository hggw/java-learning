package LeetCodeTanXin.easy;

/**
 * 135. 分发糖果
 * @author HGW
 */
public class Candy {
    public int candy(int[] ratings) {
        int length = ratings.length;
        int[] res = new int[length];
        res[0] = 1;
        for(int i = 1; i < length; i++) {
            res[i] = ratings[i] > ratings[i-1] ? res[i-1] + 1 : 1;
        }
        int right = 0;
        int ans = 0;
        for(int i = length - 1; i >= 0 ; i--) {
            right =  i < length - 1 && ratings[i] > ratings[i+1] ? right + 1: 1;
            ans += Math.max(right,res[i]);
        }
        return ans;
    }
}
