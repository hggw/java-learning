package TestDemo;

public class test {
    public static void main(String[] args) {
        // 2010,2011,2012,2013
        int[] data = new int[]{31,28,31,30,31,30,31,31,30,31,30,31,
                              31,28,31,30,31,30,31,31,30,31,30,31,
                              31,29,31,30,31,30,31,31,30,31,30,31,
                              31,28,31,30,31,30,31,31,30,31,30,31};
        int[] result = new int[36];
        int index = 0;
        for(int i = 0; i < 34; i++) {
            result[index++] = data[i]+data[i+1]+data[i+1];
        }





    }
    public static int[] kReflect(int n, int m , int k) {
        int count = 0;
        int next = 0;
        k = k % (2*m+2*n-2);
        while(count < k) {
            count++;
            if(count <= m+n-1) {
                if (count % 2 == 1) {
                    next = Math.abs(2 * m - next);
                } else {
                    next = Math.abs(2 * m + 2 * n - next);
                }
            } else {
                if (count % 2 == 0) {
                    next = Math.abs(2 * m - next);
                } else {
                    next = Math.abs(2 * m + 2 * n - next);
                }
            }
        }
        
        int[] res = new int[2];
        if(next <= m ) {
            res[0] = next;
            res[1] = 0;
        } else if(next <= n+m) {
            res[0] = m;
            res[1] = next - m;
        } else if(next <= 2*m+n) {
            res[0] = m-(next-n-m);
            res[1] = n;
        } else {
            res[0] = 0;
            res[1] = next - (2*m+n);
        }
        return res;
    }


    static int[] dr = new int[]{1,1,1};
    static int[] dc = new int[]{-1,0,1};
    public static int minFallingPathSum(int[][] matrix) {
        int ans = Integer.MAX_VALUE;
        for(int i = 0; i < matrix[0].length;i++) {

            ans = Math.min(ans,dfs(matrix,0,i));
        }
        return ans;
    }

    private static int dfs(int[][] matrix, int sr,int sc) {
        int n = matrix.length, m = matrix[0].length;
        if(sr < 0 || sr >= n || sc < 0 || sc >= m) {
            return Integer.MAX_VALUE;
        }
        int minDist = matrix[sr][sc];
        int dist = Integer.MAX_VALUE;
        for(int i = 0; i < 3; i++) {
            int x = sr+dr[i], y = sc + dc[i];
            if(x < n) {
                dist = Math.min(dist,dfs(matrix,x,y));
            }
        }
        return minDist + dist;

    }
}
