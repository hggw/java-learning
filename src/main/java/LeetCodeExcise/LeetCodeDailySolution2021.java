package LeetCodeExcise;

import java.util.*;

public class LeetCodeDailySolution2021 {
    public static void main(String[] args) {
        int[] nums1 = new int[]{1,1,10};

        int res = minMoves(nums1);
    }

    /**
     * 453. 最小操作次数使数组元素相等
     * @param nums
     * @return
     * [1,1,1000000000]
     */
    public static int minMoves(int[] nums) {
        int minNum = Arrays.stream(nums).min().getAsInt();
        int res = 0;
        for (int num : nums) {
            res += num - minNum;
        }
        return res;
    }

    /**
     * 1893. 检查是否区域内所有整数都被覆盖
     * @param ranges
     * @param left
     * @param right
     * @return
     */


    /**
     * 剑指 Offer 53 - I. 在排序数组中查找数字 I
     * @param nums
     * @param target
     * @return
     */
    public int search1(int[] nums, int target) {
        int leftIdx = binarySearch(nums, target, true);
        int rightIdx = binarySearch(nums, target, false) - 1;
        if (leftIdx <= rightIdx && rightIdx < nums.length && nums[leftIdx] == target && nums[rightIdx] == target) {
            return rightIdx - leftIdx + 1;
        }
        return 0;
    }

    public int binarySearch(int[] nums, int target, boolean lower) {
        int left = 0, right = nums.length - 1, ans = nums.length;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] > target || (lower && nums[mid] >= target)) {
                right = mid - 1;
                ans = mid;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }

    /** ##
     * 218. 天际线问题
     * @param buildings
     * @return
     */
    public List<List<Integer>> getSkyline(int[][] buildings) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> b[1] - a[1]);
        List<Integer> boundaries = new ArrayList<>();
        for (int[] building : buildings) {
            boundaries.add(building[0]);
            boundaries.add(building[1]);
        }
        Collections.sort(boundaries);

        List<List<Integer>> ret = new ArrayList<>();
        int n = buildings.length, idx = 0;
        for (int boundary : boundaries) {
            while (idx < n && buildings[idx][0] <= boundary) {
                pq.offer(new int[]{buildings[idx][1], buildings[idx][2]});
                idx++;
            }
            while (!pq.isEmpty() && pq.peek()[0] <= boundary) {
                pq.poll();
            }

            int maxn = pq.isEmpty() ? 0 : pq.peek()[1];
            if (ret.size() == 0 || maxn != ret.get(ret.size() - 1).get(1)) {
                ret.add(Arrays.asList(boundary, maxn));
            }
        }
        return ret;
    }

    /**
     * 275. H 指数 II
     * @param citations
     * @return
     */
    public int hIndex1(int[] citations) {
        int length = citations.length;
        int left = 0, right = length - 1;
        while(left <= right) {
            int mid = (left + right) / 2;
            if(citations[mid] >= length - mid) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return length - left;
    }

    /**
     * 274. H 指数
     * [1,2,3,4,5,6,7]
     * @param citations
     * @return
     */
    public int hIndex(int[] citations) {
        Arrays.sort(citations);
        int length = citations.length;
        int high = 0;
        int i = length - 1;
        while(i >= 0 && citations[i] > high) {
            high++;
            i--;
        }
        return high;
    }

    /** 感觉这可以应用在微服务的主从选举机制上
     * 面试题 17.10. 主要元素
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        int candidate = -1;
        int count = 0;
        for (int num : nums) {
            if (count == 0) {
                candidate = num;
            }
            if (num == candidate) {
                count++;
            } else {
                count--;
            }
        }
        count = 0;
        int length = nums.length;
        for (int num : nums) {
            if (num == candidate) {
                count++;
            }
        }
        return count * 2 > length ? candidate : -1;
    }

    /**
     * 930. 和相同的二元子数组
     * @param nums
     * @param goal
     * @return
     */
    public int numSubarraysWithSum(int[] nums, int goal) {
        int length = nums.length;
        int left1 = 0, left2 = 0, right = 0;
        int sum1 = 0, sum2 = 0;
        int ret = 0;
        while (right < length) {
            sum1 += nums[right];
            while (left1 <= right && sum1 > goal) {
                sum1 -= nums[left1];
                left1++;
            }
            sum2 += nums[right];
            while (left2 <= right && sum2 >= goal) {
                sum2 -= nums[left2];
                left2++;
            }
            ret += left2 - left1;
            right++;
        }
        return ret;
    }

    /**
     * 1418. 点菜展示表
     * @param orders
     * @return
     */
    public List<List<String>> displayTable(List<List<String>> orders) {
        // 从订单中获取餐品名称和桌号，统计每桌点餐数量
        Set<String> nameSet = new HashSet<String>();
        Map<Integer, Map<String, Integer>> foodsCnt = new HashMap<Integer, Map<String, Integer>>();
        for (List<String> order : orders) {
            nameSet.add(order.get(2));
            int id = Integer.parseInt(order.get(1));
            Map<String, Integer> map = foodsCnt.getOrDefault(id, new HashMap<String, Integer>());
            map.put(order.get(2), map.getOrDefault(order.get(2), 0) + 1);
            foodsCnt.put(id, map);
        }

        // 提取餐品名称，并按字母顺序排列
        int n = nameSet.size();
        List<String> names = new ArrayList<String>();
        for (String name : nameSet) {
            names.add(name);
        }
        Collections.sort(names);
        // 填写点菜展示表
        List<List<String>> table = new ArrayList<List<String>>();
        List<String> header = new ArrayList<String>();
        header.add("Table");
        for (String name : names) {
            header.add(name);
        }
        table.add(header);

        // 提取桌号，并按餐桌桌号升序排列
        int m = foodsCnt.size();
        List<Integer> ids = new ArrayList<Integer>();
        for (int id : foodsCnt.keySet()) {
            ids.add(id);
        }
        Collections.sort(ids);


        for (int i = 0; i < m; ++i) {
            int id = ids.get(i);
            Map<String, Integer> cnt = foodsCnt.get(id);
            List<String> row = new ArrayList<String>();
            row.add(Integer.toString(id));
            for (int j = 0; j < n; ++j) {
                row.add(Integer.toString(cnt.getOrDefault(names.get(j), 0)));
            }
            table.add(row);
        }
        return table;
    }


    /** ##
     * 剑指 Offer 37. 序列化二叉树
     * @param root
     * @return
     */
    public String serialize(TreeNode root) {
        if (root == null) {
            return "X";
        }
        String left = "(" + serialize(root.left) + ")";
        String right = "(" + serialize(root.right) + ")";
        return left + root.val + right;
    }

    public TreeNode deserialize(String data) {
        int[] ptr = {0};
        return parse(data, ptr);
    }

    public TreeNode parse(String data, int[] ptr) {
        if (data.charAt(ptr[0]) == 'X') {
            ++ptr[0];
            return null;
        }
        TreeNode cur = new TreeNode(0);
        cur.left = parseSubtree(data, ptr);
        cur.val = parseInt(data, ptr);
        cur.right = parseSubtree(data, ptr);
        return cur;
    }

    public TreeNode parseSubtree(String data, int[] ptr) {
        ++ptr[0]; // 跳过左括号
        TreeNode subtree = parse(data, ptr);
        ++ptr[0]; // 跳过右括号
        return subtree;
    }

    public int parseInt(String data, int[] ptr) {
        int x = 0, sgn = 1;
        if (!Character.isDigit(data.charAt(ptr[0]))) {
            sgn = -1;
            ++ptr[0];
        }
        while (Character.isDigit(data.charAt(ptr[0]))) {
            x = x * 10 + data.charAt(ptr[0]++) - '0';
        }
        return x * sgn;
    }

    /**
     * 168. Excel表列名称
     * @param columnNumber
     * @return
     */
    public static String convertToTitle(int columnNumber) {
        StringBuilder res = new StringBuilder();
        while(columnNumber != 0) {
            columnNumber--;
            res.append((char)('A'+ columnNumber%26));
            columnNumber /= 26;
        }
        return res.reverse().toString();
    }

    /**
     * 剑指 Offer 15. 二进制中1的个数
     * @param n
     * @return
     */
    public static int hammingWeight(int n) {
        int count = 0;
        int tmp = 1;
        for(int i = 0; i < 32; i++) {
            if((n & tmp) != 0) {
                count++;
            }
            tmp <<= 1;
        }
        return count;
    }

    /**
     * 401. 二进制手表
     * @param turnedOn
     * @return
     */
    public List<String> readBinaryWatch(int turnedOn) {
        List<String> ans = new ArrayList<>();
        for(int i = 0; i < 1024; i++) {
            int h = i >> 6, m = i & 63;
            if(h < 12 && m < 60 && Integer.bitCount(i) == turnedOn) {
                ans.add(h + ":" + (m < 10 ? "0" : "") + m);
            }
        }
        return ans;
    }

    /**
     * 483. 最小好进制
     * @param n
     * @return
     */
    public String smallestGoodBase(String n) {
        long nVal = Long.parseLong(n);
        int mMax = (int) Math.floor(Math.log(nVal) / Math.log(2));
        for (int m = mMax; m > 1; m--) {
            int k = (int) Math.pow(nVal, 1.0 / m);
            long mul = 1, sum = 1;
            for (int i = 0; i < m; i++) {
                mul *= k;
                sum += mul;
            }
            if (sum == nVal) {
                return Integer.toString(k);
            }
        }
        return Long.toString(nVal - 1);
    }

    /**
     * 877. 石子游戏
     * @param piles
     * @return
     */
    public boolean stoneGame(int[] piles) {
        int length = piles.length;
        int[][] dp = new int[length][length];
        for(int i = 0; i < length; i++) {
            dp[i][i] = piles[i];
        }
        for(int i = length - 2; i >= 0; i--) {
            for(int j = i + 1; j < length; j++) {
                dp[i][j] = Math.max(piles[i] - dp[i + 1][j],piles[j] - dp[i][j - 1]);
            }
        }
        return dp[0][length - 1] > 0;
    }

    /** ##
     * 1049. 最后一块石头的重量 II
     * @param stones
     * @return
     */
    public int lastStoneWeightII(int[] stones) {
        int sum = 0;
        for (int weight : stones) {
            sum += weight;
        }
        int m = sum / 2;
        boolean[] dp = new boolean[m + 1];
        dp[0] = true;
        for (int weight : stones) {
            for (int j = m; j >= weight; --j) {
                dp[j] = dp[j] || dp[j - weight];
            }
        }
        for (int j = m;; --j) {
            if (dp[j]) {
                return sum - 2 * j;
            }
        }
    }


    /**
     * 810. 黑板异或游戏
     * @param nums
     * @return
     */
    public boolean xorGame(int[] nums) {
        if (nums.length % 2 == 0) {
            return true;
        }
        int xor = 0;
        for (int num : nums) {
            xor ^= num;
        }
        return xor == 0;
    }

    /**
     * 1035. 不相交的线
     * @param nums1
     * @param nums2
     * @return
     */
    public static int maxUncrossedLines(int[] nums1, int[] nums2) {
        int length1 = nums1.length + 1;
        int length2 = nums2.length + 1;
        int[][] dp = new int[length1][length2];
        for(int i = 1; i < length1; i++) {
            int num1 = nums1[i - 1];
            for(int j = 1; j < length2; j++) {
                int num2 = nums2[j - 1];
                if(num1 == num2) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j],dp[i][j - 1]);
                }
            }
        }
        return dp[length1 - 1][length2 - 1];
    }

    /**
     * 692. 前K个高频单词
     * @param words
     * @param k
     * @return
     */
    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> cnt = new HashMap<String, Integer>();
        for (String word : words) {
            cnt.put(word, cnt.getOrDefault(word, 0) + 1);
        }
        List<String> rec = new ArrayList<String>();
        for (Map.Entry<String, Integer> entry : cnt.entrySet()) {
            rec.add(entry.getKey());
        }
        Collections.sort(rec, new Comparator<String>() {
            @Override
            public int compare(String word1, String word2) {
                return cnt.get(word1).equals(cnt.get(word2)) ? word1.compareTo(word2) : cnt.get(word2) - cnt.get(word1);
            }
        });
        return rec.subList(0, k);
    }

    /**
     * 1738. 找出第 K 大的异或坐标值
     * @param matrix
     * @param k
     * @return
     */
    public int kthLargestValue(int[][] matrix, int k) {
        int row = matrix.length, col = matrix[0].length;
        int[][] data = new int[row+1][col+1];
        List<Integer> res = new ArrayList<>();
        for(int i = 1; i < row+1; i++) {
            for(int j = 1; j < col+1; j++) {
                data[i][j] = data[i-1][j]^data[i][j-1]^data[i-1][j-1]^matrix[i-1][j-1];
                res.add(data[i][j]);
            }
        }
        Collections.sort(res,(o1,o2) -> o2 - o1);
        return res.get(k-1);
    }

    /**
     * 1442. 形成两个异或相等数组的三元组数目
     * @param arr
     * @return
     */
    public int countTriplets(int[] arr) {
        int length = arr.length;
        int count = 0;
        int[] data = new int[length + 1];
        data[0] = 0;
        for(int i = 0; i < length; i++) {
            data[i+1] = data[i]^arr[i];
        }
        for(int i = 0; i < length; i++) {
            for(int j = i + 1; j < length; j++) {
                if(data[i] == data[j + 1]) {
                    count += j - i;
                }
            }
        }
        return count;
    }

    /** ##
     * 993. 二叉树的堂兄弟节点
     * @param root
     * @param x
     * @param y
     * @return
     */
    // x 的信息
    int x;
    TreeNode xParent;
    int xDepth;
    boolean xFound = false;

    // y 的信息
    int y;
    TreeNode yParent;
    int yDepth;
    boolean yFound = false;

    public boolean isCousins(TreeNode root, int x, int y) {
        this.x = x;
        this.y = y;

        Queue<TreeNode> nodeQueue = new LinkedList<TreeNode>();
        Queue<Integer> depthQueue = new LinkedList<Integer>();
        nodeQueue.offer(root);
        depthQueue.offer(0);
        update(root, null, 0);

        while (!nodeQueue.isEmpty()) {
            TreeNode node = nodeQueue.poll();
            int depth = depthQueue.poll();
            if (node.left != null) {
                nodeQueue.offer(node.left);
                depthQueue.offer(depth + 1);
                update(node.left, node, depth + 1);
            }
            if (node.right != null) {
                nodeQueue.offer(node.right);
                depthQueue.offer(depth + 1);
                update(node.right, node, depth + 1);
            }
            if (xFound && yFound) {
                break;
            }
        }

        return xDepth == yDepth && xParent != yParent;
    }

    // 用来判断是否遍历到 x 或 y 的辅助函数
    public void update(TreeNode node, TreeNode parent, int depth) {
        if (node.val == x) {
            xParent = parent;
            xDepth = depth;
            xFound = true;
        } else if (node.val == y) {
            yParent = parent;
            yDepth = depth;
            yFound = true;
        }
    }

    /**
     * 203. 移除链表元素
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val) {
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode pre = res, curr = res.next;
        while(curr != null) {
            if(curr.val == val) {
                pre.next = curr.next;
            } else {
                pre = curr;
            }
            curr = curr.next;
        }
        return res.next;
    }

    /**
     * 12. 整数转罗马数字
     * @param num
     * @return
     */
    public static String intToRoman(int num) {
        String[] thousands = new String[]{"","M","MM","MMM"};
        String[] hundreds = new String[]{"","C","CC","CCC","CD","D","DC","DCC","DCCC","CM"};
        String[] tens = new String[]{"","X","XX","XXX","XL","L","LX","LXX","LXXX","XC"};
        String[] ones = new String[]{"","I","II","III","IV","V","VI","VII","VIII","IX"};
        StringBuilder res = new StringBuilder();
        res.append(thousands[num/1000]);
        res.append(hundreds[num%1000/100]);
        res.append(tens[num%100/10]);
        res.append(ones[num%10]);
        return res.toString();

    }

    /**
     * 1269. 停在原地的方案数
     * @param steps
     * @param arrLen
     * @return
     */
    public static int numWays(int steps, int arrLen) {
        final int model = 1000000007;
        int distance = Math.min(arrLen - 1, steps);
        int[][] dp = new int[steps + 1][distance + 1];
        dp[0][0] = 1;
        for(int i = 1; i < steps + 1; i++) {
            for(int j = 0; j < distance + 1; j++) {
                dp[i][j] = dp[i - 1][j] % model;
                if(j - 1 >= 0) {
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % model;
                }
                if(j + 1 <= distance) {
                    dp[i][j] = (dp[i][j] + dp[i - 1][j + 1]) % model;
                }
            }
        }
        return dp[steps][0];

    }

    /**
     * 1310. 子数组异或查询
     * @param arr
     * @param queries
     * @return
     */
    public int[] xorQueries(int[] arr, int[][] queries) {
        int length = queries.length;
        int len = arr.length;
        int[] data = new int[len + 1];
        for(int i = 0; i < len; i++) {
            data[i + 1] = data[i]^arr[i];
        }
        int[] res = new int[length];
        for(int i = 0; i < length; i++) {
            res[i] = data[queries[i][0]]^data[queries[i][1]];
        }
        return res;

    }

    /**
     * 1734. 解码异或后的排列
     * @param encoded
     * @return
     */
    public static int[] decode(int[] encoded) {
        int length = encoded.length + 1;
        int total = 0;
        for(int i = 1; i <= length; i++) {
            total ^= i;
        }
        int odd = 0;
        for(int i = 1; i < length - 1; i += 2) {
            odd ^= encoded[i];
        }
        int[] data = new int[length];
        data[0] = odd^total;
        for(int i = 1; i <= length - 1; i++) {
            data[i] = encoded[i - 1]^data[i - 1];
        }
        return data;
    }


    /**
     * 872. 叶子相似的树
     * @param root1
     * @param root2
     * @return
     */
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> data1 = new ArrayList<>();
        List<Integer> data2 = new ArrayList<>();
        if(root1 != null) {
            dfs(root1,data1);
        }
        if(root2 != null) {
            dfs(root2,data2);
        }
        return data1.equals(data2);

    }
    public void dfs(TreeNode root,List<Integer> data) {
        if(root.left == null && root.right == null) {
            data.add(root.val);
        } else {
            if(root.left != null) {
                dfs(root.left, data);
            }
            if(root.right != null) {
                dfs(root.right, data);
            }
        }
    }

    /** ##还没来得及看
     * 1723. 完成所有工作的最短时间
     * @param jobs
     * @param k
     * @return
     */
    public int minimumTimeRequired(int[] jobs, int k) {
        Arrays.sort(jobs);
        int low = 0, high = jobs.length - 1;
        while (low < high) {
            int temp = jobs[low];
            jobs[low] = jobs[high];
            jobs[high] = temp;
            low++;
            high--;
        }
        int l = jobs[0], r = Arrays.stream(jobs).sum();
        while (l < r) {
            int mid = (l + r) >> 1;
            if (check(jobs, k, mid)) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    public boolean check(int[] jobs, int k, int limit) {
        int[] workloads = new int[k];
        return backtrack(jobs, workloads, 0, limit);
    }

    public boolean backtrack(int[] jobs, int[] workloads, int i, int limit) {
        if (i >= jobs.length) {
            return true;
        }
        int cur = jobs[i];
        for (int j = 0; j < workloads.length; ++j) {
            if (workloads[j] + cur <= limit) {
                workloads[j] += cur;
                if (backtrack(jobs, workloads, i + 1, limit)) {
                    return true;
                }
                workloads[j] -= cur;
            }
            // 如果当前工人未被分配工作，那么下一个工人也必然未被分配工作
            // 或者当前工作恰能使该工人的工作量达到了上限
            // 这两种情况下我们无需尝试继续分配工作
            if (workloads[j] == 0 || workloads[j] + cur == limit) {
                break;
            }
        }
        return false;
    }

    /**
     * 1720. 解码异或后的数组
     * @param encoded
     * @param first
     * @return
     */
    public static int[] decode(int[] encoded, int first) {
        int length = encoded.length;
        int[] res = new int[length + 1];
        res[0] = first;
        for(int i = 0; i < length; i++) {
            res[i + 1] = res[i]^encoded[i];
        }
        return res;
    }

    /** ##
     * 137. 只出现一次的数字 II
     * 思路：骚套路
     * ~:按位非
     * &：与
     * |：或
     * ^：异或
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        int a = 0, b = 0;
        for (int num : nums) {
            b = ~a & (b ^ num);
            a = ~b & (a ^ num);
        }
        return b;
    }

    /** ##
     * 403. 青蛙过河
     * 思路未整理
     * @param stones
     * @return
     */
    public static boolean canCross(int[] stones) {
        int length = stones.length;
        boolean[][] dp = new boolean[length][length];
        dp[0][0] = true;
        for (int i = 1; i < length; ++i) {
            if (stones[i] - stones[i - 1] > i) {
                return false;
            }
        }
        for (int i = 1; i < length; ++i) {
            for (int j = i - 1; j >= 0; --j) {
                int k = stones[i] - stones[j];
                if (k > j + 1) {
                    break;
                }
                dp[i][k] = dp[j][k - 1] || dp[j][k] || dp[j][k + 1];
                if (i == length - 1 && dp[i][k]) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 633. 平方数之和
     * @param c
     * @return
     */
    public static boolean judgeSquareSum(int c) {
        int left = 0, right = (int) Math.pow(c,0.5);
        while(left <= right) {
            int sum = (int) (Math.pow(left,2) + Math.pow(right,2));
            if(sum == c) {
                return true;
            } else if (sum > c){
                right--;
            } else {
                left++;
            }
        }
        return false;
    }

    /**
     * 14. 最长公共前缀
     * @param strs
     * @return
     */
    public String longestCommonPrefix(String[] strs) {
        String pre = strs[0];
        for(String str : strs) {
            pre = longestPre(pre,str);
        }
        return pre;
    }
    public String longestPre(String pre,String str) {
        int length = Math.min(pre.length(),str.length());
        int index = 0;
        for(int i = 0; i < length; i++) {
            if(pre.charAt(i) == str.charAt(i)) {
                index++;
            } else {
                break;
            }
        }
        return pre.substring(0,index);
    }

    /**
     * 938. 二叉搜索树的范围和
     * 思路：1、根据节点值大小缩小范围，递归取进行搜索
     * @param root
     * @param L
     * @param R
     * @return
     */
    public int rangeSumBST1(TreeNode root, int L, int R) {
        if(root == null) {
            return 0;
        }
        List<Integer> data = new ArrayList<>();
        inorder1(root,data);
        int res = 0;
        for(Integer num : data) {
            if(num >= L && num <= R) {
                res += num;
            }
        }
        return res;
    }
    public int rangeSumBST2(TreeNode root, int L, int R) {
        if(root == null) {
            return 0;
        }
        int res = 0;
        if(root.val >= L && root.val <= R) {
            res += root.val + rangeSumBST2(root.left,L,R) + rangeSumBST2(root.right,L,R);
        } else if(root.val < L) {
            res += rangeSumBST2(root.right,L,R);
        } else {
            res += rangeSumBST2(root.left,L,R);
        }
        return res;
    }
    public void inorder1(TreeNode node,List<Integer> data) {
        if(node == null) {
            return;
        }
        inorder1(node.left,data);
        data.add(node.val);
        inorder1(node.right,data);
    }

    /**
     * 201.n的阶乘尾部有多少个0
     * @param n
     * @return
     */
    public int trailingZeroes(int n) {
        int count = 0,tmp = 5;
        while(n >= tmp) {
            count += (n / tmp);
            tmp *= 5;
        }
        return count;
    }

    /**
     * 1011. 在 D 天内送达包裹的能力
     * 思路：1、不考虑D的情况下，确认运载能力的范围，即weights数组里面的最大值（左边界）和数组和（有边界），满足要求的必然位于这个取值区间内
     *      2、通过二分查找，判断中间值是否满足D天内送达，不断缩小满足条件的取值范围，得到结果
     * @param weights
     * @param D
     * @return
     */
    public static int shipWithinDays(int[] weights, int D) {
        int left = 0, right = 0;
        for(int weight : weights) {
            left = Math.max(left,weight);
            right += weight;
        }
        while(left < right) {
            int mid = (left + right) / 2;
            int count = 0, tmp = 0;
            for(int weight : weights) {
                tmp += weight;
                if(tmp > mid) {
                    count++;
                    tmp = weight;
                }
            }
            count++;
            if(count <= D) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    /**
     * 897. 递增顺序搜索树
     * @param root
     * @return
     */
    public TreeNode increasingBST(TreeNode root) {
        List<Integer> data = new ArrayList<>();
        inorder(root,data);
        TreeNode ans = new TreeNode(-1);
        TreeNode tmp = ans;
        for(Integer num : data) {
            tmp.right = new TreeNode(num);
            tmp = tmp.right;
        }
        return ans.right;
    }
    public void inorder(TreeNode node ,List<Integer> data) {
        if(node == null) {
            return;
        }
        inorder(node.left,data);
        data.add(node.val);
        inorder(node.right,data);
    }

    /**
     * 377. 组合总和 Ⅳ
     * @param nums
     * @param target
     * @return
     */
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for(int i = 1; i <= target; i++) {
            for(int num : nums) {
                if(num < i) {
                    dp[i] += dp[i - num];
                }
            }
        }
        return dp[target];
    }

    /**
     * 368. 最大整除子集
     * 思路：1、将nums非递减排序，通过遍历得到数组中每个元素对应的整除数的个数（存在dp数组中）
     *      2、得到dp数组中的最大值maxSize和对应的最大数组元素maxValue，即为最大整除子集的长度
     *      3、对排序后的nums数组进行倒序遍历，从dp最大值开始倒序遍历，每次遍历一个对应元素，maxSize减1，maxValue赋值为当前的整除数
     * @param nums
     * @return
     */
    public List<Integer> largestDivisibleSubset(int[] nums) {
        int length = nums.length;
        Arrays.sort(nums);
        int[] dp = new int[length];
        Arrays.fill(dp,1);
        int maxSize = 1, maxValue = dp[0];
        for(int i = 1; i < length; i++) {
            for(int j = 0; j < i; j++) {
                if(nums[i] % nums[j] == 0) {
                    dp[i] = Math.max(dp[i],dp[j] + 1);
                }
            }
            if(dp[i] > maxSize) {
                maxSize = dp[i];
                maxValue = nums[i];
            }
        }
        List<Integer> res = new ArrayList<>();
        if(maxSize == 1) {
            res.add(nums[0]);
        } else {
            for(int i = length - 1; i >= 0 && maxValue > 0; i--) {
                if(dp[i] == maxSize && maxValue % nums[i] == 0) {
                    res.add(nums[i]);
                    maxSize--;
                    maxValue = nums[i];
                }
            }
        }
        return res;
    }

    /**
     * 363. 矩形区域不超过 K 的最大数值和
     * 算法思路：1、通过遍历矩阵中不相同的两行(i,j)，分别计算出这两个行号之间的单列和
     *          2、将所得到的列和做差，即可得到已知两行(a,b)之间，位于这两列之间的所有元素和（即为左上角为[i,a],右上角为[j,b]的子矩阵的元素和）
     * @param matrix
     * @param k
     * @return
     */
    public static int maxSumSubmatrix(int[][] matrix, int k) {
        int row = matrix.length, col = matrix[0].length;
        int ans = Integer.MIN_VALUE;
        for(int i = 0; i < row; i++) {
            int[] data = new int[col];
            for(int j = i; j < row; j++) {
                for(int l = 0; l < col; l++) {
                    data[l] += matrix[j][l];
                }
                TreeSet<Integer> dataSet = new TreeSet<>();
                dataSet.add(0);
                int sum = 0;
                for(int num : data) {
                    sum += num;
                    Integer tmp = dataSet.ceiling(sum - k);
                    if(tmp != null) {
                        ans = Math.max(ans, sum - tmp);
                    }
                    dataSet.add(sum);
                }
            }
        }
        return ans;
    }

    /**
     * 91. 解码方法
     * @param s
     * @return
     */
    public static int numDecodings(String s) {
        int length = s.length();
        int[] res = new int[length + 1];
        res[0] = 1;
        for(int i = 1; i <= length; i++) {
            if(s.charAt(i - 1) != '0') {
                res[i] += res[i - 1];
            }
            if(i > 1 && s.charAt(i - 2) != '0' && ((s.charAt(i - 2) - '0') * 10 + s.charAt(i - 1) - '0') <= 26) {
                res[i] += res[i - 2];
            }
        }
        return res[length];
    }

    /** KMP
     * 28. 实现 strStr()
     * @param haystack
     * @param needle
     * @return int
     */
    public static int strStr(String haystack, String needle) {
        int n = haystack.length(), m = needle.length();
        if (m == 0) return 0;
        int[] pi = new int[m];
        for (int i = 1, j = 0; i < m; i++) {
            while (j > 0 && needle.charAt(i) != needle.charAt(j)) {
                j = pi[j - 1];
            }
            if (needle.charAt(i) == needle.charAt(j)) {
                j++;
            }
            pi[i] = j;
        }
        for (int i = 0, j = 0; i < n; i++) {
            while (j > 0 && haystack.charAt(i) != needle.charAt(j)) {
                j = pi[j - 1];
            }
            if (haystack.charAt(i) == needle.charAt(j)) {
                j++;
            }
            if (j == m) {
                return i - m + 1;
            }
        }
        return -1;

    }

    /**
     * 27. 移除元素
     * @param nums
     * @param val
     * @return
     */
    public int removeElement(int[] nums, int val) {
        int length = nums.length;
        int left = 0;
        for(int right = 0; right < length; right++) {
            if(nums[right] != val) {
                nums[left++] = nums[right];
            }
        }
        return left;
    }

    /**
     * 213. 打家劫舍 II
     * @param nums
     * @return
     */
    private static int rob(int[] nums) {
        int length = nums.length;
        if(length == 1) return nums[0];
        if(length == 2) return Math.max(nums[0],nums[1]);
        int first1 = nums[0],second1 = Math.max(nums[0],nums[1]);
        for(int i = 2; i < length - 1; i++) {
            int tmp1 = second1;
            second1 = Math.max(first1 + nums[i],second1);
            first1 = tmp1;
        }
        int first2 = nums[1],second2 = Math.max(nums[1],nums[2]);
        for(int i = 3; i < length; i ++) {
            int tmp2 = second2;
            second2 = Math.max(first2 + nums[i],second2);
            first2 = tmp2;
        }
        return Math.max(second1,second2);
    }

    /**
     * 783. 二叉搜索树节点最小距离
     * @param root
     * @return
     */
    int pre;
    int ans;
    public int minDiffInBST(TreeNode root) {
        ans = Integer.MAX_VALUE;
        pre = -1;
        dfs(root);
        return ans;
    }
    public void dfs(TreeNode root) {
        if(root == null) {
            return;
        }
        dfs(root.left);
        if(pre != -1) {
            ans = Math.min(ans,root.val - pre);
        }
        pre = root.val;
        dfs(root.right);
    }

    /**
     * 179. 最大数
     * @param nums
     * @return
     */
    public String largestNumber(int[] nums) {
        int length = nums.length;
        Integer[] data = new Integer[length];
        for(int i = 0; i < length; i++) {
            data[i] = nums[i];
        }
        Arrays.sort(data,(x,y) -> {
            long sx = 10, sy = 10;
            while(sx <= x) {
                sx *= 10;
            }
            while(sy <= y) {
                sy *= 10;
            }
            return (int) (-sy * x - y + sx * y + x);
        });
        if(data[0] == 0) {
            return "0";
        }
        StringBuilder res = new StringBuilder();
        for(Integer num : data) {
            res.append(num);
        }
        return res.toString();
    }

    /**
     * 5726. 数组元素积的符号
     * @param nums
     * @return
     */
    public int arraySign(int[] nums) {
        Map<Integer,Integer> data = new HashMap<>();
        int count1 = 0, count2 = 0, count3= 0;
        for(int num : nums) {
            if(num > 0) {
                count1++;
            } else if(num < 0) {
                count2++;
            } else {
                count3++;
            }
        }
        if(count3 != 0) {
            return 0;
        } else if(count1 * count2 != 0) {
            if(count2 % 2 != 0) {
                return -1;
            } else {
                return 1;
            }
        } else {
            if(count1 == 0) {
                if(count2 % 2 != 0) {
                    return - 1;
                } else {
                    return 1;
                }
            } else {
                return 1;
            }
        }
    }

    /**
     * 154. 寻找旋转排序数组中的最小值 II(存在 重复 元素值的数组 nums)
     * @param nums
     * @return
     */
    public static int findMin(int[] nums) {
        int length = nums.length;
        int left =  0, right = length - 1;
        while(left < right) {
            int mid = (left + right)/2;
            if(nums[left] <= nums[mid]) {
                if (nums[mid] < nums[right]) {
                    right = mid;
                } else if (nums[mid] == nums[right]){
                    right--;
                } else {
                    left = mid + 1;
                }
            } else {
                right = mid;
            }
        }
        return nums[left];
    }

    /**
     * 153. 寻找旋转排序数组中的最小值
     * @param nums
     * @return
     */
    public int findMin1(int[] nums) {
        int length = nums.length;
        int left =  0, right = length - 1;
        while(left < right) {
            int mid = (left + right)/2;
            if(nums[left] <= nums[mid]) {
                if(nums[mid] <= nums[right]) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            } else {
                right = mid;
            }
        }
        return nums[left];
    }

    /**
     * 81. 搜索旋转排序数组 II
     * @param nums
     * @param target
     * @return
     */
    public boolean search(int[] nums, int target) {
        int length = nums.length;
        if(length == 0 ) return false;
        if(length == 1) return nums[0] == target;
        int left = 0, right = length - 1;
        while(left <= right) {
            int mid = (left + right)/2;
            if(nums[mid] == target) {
                return true;
            }
            if(nums[left] == nums[mid] && nums[mid] == nums[right]) {
                left++;
                right--;
            } else if(nums[left] <= nums[mid]) {
                if(nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if(nums[mid] < target && target <= nums[length - 1]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return false;
    }

    /**
     * 80. 删除有序数组中的重复项 II
     * @param nums
     * @return
     */
    public static int removeDuplicates(int[] nums) {
        int length = nums.length;
        if(length <= 2) return length;
        int slow = 2, fast = 2;
        for(;fast < length; fast++) {
            if(nums[slow - 2] != nums[fast]) {
                nums[slow++] = nums[fast];
            }
        }
        return slow;
    }

    /**
     * 781. 森林中的兔子
     * @param answers
     * @return
     */
    public int numRabbits(int[] answers) {
        Map<Integer, Integer> count = new HashMap<>();
        for (int y : answers) {
            count.put(y, count.getOrDefault(y, 0) + 1);
        }
        int ans = 0;
        for (Map.Entry<Integer, Integer> entry : count.entrySet()) {
            int y = entry.getKey(), x = entry.getValue();
            ans += (x + y) / (y + 1) * (y + 1);
        }
        return ans;
    }

    /**
     * 88. 合并两个有序数组
     * @param nums1
     * @param m
     * @param nums2
     * @param n
     */
    public static void merge(int[] nums1, int m, int[] nums2, int n) {
        int index1 = m - n - 1, index2 = n - 1;
        int index = m - 1;
        while(index1 >= 0 && index2 >= 0){
            if(nums1[index1] >= nums2[index2]) {
                nums1[index--] = nums1[index1--];
            } else {
                nums1[index--] = nums2[index2--];
            }
        }
        while (index1 >= 0) {
            nums1[index--] = nums1[index1--];
        }
        while (index2 >= 0) {
            nums1[index--] = nums2[index2--];
        }
    }

    /**
     * 面试题 17.21. 直方图的水量
     * @param height
     * @return
     */
    public static int trap(int[] height) {
        int length = height.length;
        if(length <= 0) {
            return 0;
        }
        int[] leftMax = new int[length];
        leftMax[0] = height[0];
        int[] rightMax = new int[length];
        rightMax[length - 1] = height[length - 1];
        for(int i = 1; i < length; i++) {
            leftMax[i] = Math.max(leftMax[i - 1],height[i]);
        }
        for(int i = length - 2; i >= 0; i--) {
            rightMax[i] = Math.max(rightMax[i + 1],height[i]);
        }
        int res = 0;
        for(int i = 0; i < length; i++) {
            res += Math.min(leftMax[i],rightMax[i]) - height[i];
        }
        return res;
    }

    /**
     * 1006. 笨阶乘
     * @param N
     * @return
     */
    public static int clumsy(int N) {
        int res = 0;
        if(N == 2) return 2;
        if (N == 1) return 1;
        if(N >= 3) {
            res += (N * (N - 1) / (N - 2));
            N -= 3;
        }
        while (N > 0) {
            if(N >= 4) {
                res += N - ((N - 1) * (N - 2) /(N - 3));
                N -= 4;
            } else {
                break;
            }
        }
        if(N > 0) {
            if(N > 2) {
                res += N - (N - 1) * (N - 2);
            } else {
                res += N - (N - 1);
            }
        }
        return res;
    }

    /**
     * 90. 子集 II
     * @param nums
     * @return
     */
    List<Integer> t = new ArrayList<Integer>();
    List<List<Integer>> ans1 = new ArrayList<List<Integer>>();
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        dfs(false, 0, nums);
        return ans1;
    }
    public void dfs(boolean choosePre, int cur, int[] nums) {
        if (cur == nums.length) {
            ans1.add(new ArrayList<Integer>(t));
            return;
        }
        dfs(false, cur + 1, nums);
        if (!choosePre && cur > 0 && nums[cur - 1] == nums[cur]) {
            return;
        }
        t.add(nums[cur]);
        dfs(true, cur + 1, nums);
        t.remove(t.size() - 1);
    }

    /**
     * 74. 搜索二维矩阵
     * @param matrix
     * @param target
     * @return
     */
    public static boolean searchMatrix(int[][] matrix, int target) {
        int rows = matrix.length, cols = matrix[0].length;
        if(matrix[0][0] > target || matrix[rows - 1][cols - 1] < target) return false;
        int row1 = 0, row2 = rows - 1;
        while(row1 <= row2) {
            int tmp = (row1 + row2)/2;
            if(matrix[tmp][cols - 1] < target) {
                row1 = tmp + 1;
            } else if(matrix[tmp][cols - 1] == target) {
                return true;
            } else {
                row2 = tmp - 1;
            }
        }
        int col1 = 0, col2 = cols - 1;
        while(col1 <= col2) {
            int col = (col1 + col2)/2;
            if(matrix[row1][col] < target) {
                col1 = col + 1;
            } else if(matrix[row1][col] == target) {
                return true;
            } else {
                col2 = col - 1;
            }
        }
        return matrix[row1][col1] == target;
    }
    /**
     * 190. 颠倒二进制位
     * @param n
     * @return
     */
    public static int reverseBits(int n) {
        int res = 0;
        for(int i = 0 ; i < 32; i++) {
            res |= (n & 1) << (31 - i);
            n >>= 1;
        }
        return res;
    }

    /**
     * 83. 删除排序链表中的重复元素
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        ListNode res = head;
        while(res != null && res.next != null) {
            if(res.val == res.next.val) {
                res.next = res.next.next;
            } else {
                res = res.next;
            }
        }
        return head;
    }


}
