package LeetCodeTanXin.easy;

import java.util.ArrayList;
import java.util.List;

/**
 * 763. 划分字母区间
 * @author HGW
 */
public class PartitionLabels {

    public List<Integer> partitionLabels(String s) {
        List<Integer> res = new ArrayList<>();
        int[] last = new int[26];
        for(int i = 0; i < s.length(); i++) {
            last[s.charAt(i) - 'a'] = i;
        }
        int start = 0, end = 0;
        for(int i = 0; i < s.length(); i++) {
            end = Math.max(end,last[s.charAt(i) - 'a']);
            if(i == end) {
                res.add(end - start + 1);
                start = end + 1;
            }
        }
        return res;
    }
}
