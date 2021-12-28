package LeetCodeTanXin.easy;

import java.util.Arrays;
import java.util.Comparator;

/**
 * 435. 无重叠区间
 * @author HGW
 */
public class NonoverlappingIntervals {
    public static int eraseOverlapIntervals(int[][] intervals) {
        Arrays.sort(intervals, Comparator.comparingInt(interval -> interval[1]));
        int count = 1;
        int index = 0;
        for(int i = 1; i < intervals.length; i++) {
            if(intervals[i][0] >= intervals[index][1]) {
                count++;
                index = i;
            }
        }
        return intervals.length - count;
    }

}
