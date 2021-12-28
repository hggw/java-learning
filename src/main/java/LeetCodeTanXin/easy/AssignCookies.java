package LeetCodeTanXin.easy;

import java.util.Arrays;

/**
 * 455. 分发饼干
 * @author HGW
 */
public class AssignCookies {
    public int findContentChildren1(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int length1 = g.length, length2 = s.length;
        int index1 = 0, index2 = 0;
        int count = 0;
        while(index1 < length1 && index2 < length2) {
            if(s[index2] >= g[index1]) {
                count++;
                index1++;
                index2++;
            } else {
                index2++;
            }
        }
        return count;
    }
}
