package LeetCodeTanXin.easy;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * 452. 用最少数量的箭引爆气球
 * @author HGW
 */
public class MinNumberArrowBurstBalloons {
    public static int findMinArrowShots(int[][] points) {
        Arrays.sort(points, Comparator.comparingInt(point -> point[1]));
        int length = points.length;
        int tmp = points[0][1];
        int count = 1;
        for(int i = 1; i < length; i++) {
            if(points[i][0] > tmp) {
                count++;
                tmp = points[i][1];
            }
        }
        return count;
    }
}
