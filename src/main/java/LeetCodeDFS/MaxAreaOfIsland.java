package LeetCodeDFS;

import com.sun.org.apache.regexp.internal.RE;
import io.swagger.models.auth.In;
import sun.nio.cs.ext.ISCII91;

import java.util.Deque;
import java.util.LinkedList;
import java.util.Stack;

public class MaxAreaOfIsland {
    public static void main(String[] args) {

    }

    /**
     *
     * @param grid
     * @return
     */
    public int maxAreaOfIsland(int[][] grid) {
        int result = 0;
        int[] di = {0,0,1,-1};
        int[] dj = {1,-1,0,0};
        int rows = grid.length, cols = rows != 0 ? grid[0].length : 0;
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                int cur = 0;
                Deque<Integer> stackI = new LinkedList<>();
                Deque<Integer> stackJ = new LinkedList<>();
                stackI.push(i);
                stackJ.push(j);
                while (!stackI.isEmpty()) {
                    int curI = stackI.pop(),curJ = stackJ.pop();
                    if(curI < 0 || curI >= rows || curJ < 0 || curJ >= cols || grid[curI][curJ] != 1) {
                        continue;
                    }
                    cur++;
                    grid[curI][curJ] = 0;
                    for(int k = 0; k < 4; k++) {
                        stackI.push(curI + di[k]);
                        stackJ.push(curJ + dj[k]);
                    }
                }
                result = Math.max(result, cur);
            }
        }
        return result;
    }
}
