package LeetCodeShuangZhiZhen;

/**
 * 88. 合并两个有序数组
 */
public class MergeSortedArray {
    public static void main(String[] args) {
        int[] nums1 = new int[]{2,0};
        int[] nums2 = new int[]{1};
        merge(nums1,1,nums2,1);
    }
    public static void merge(int[] nums1, int m, int[] nums2, int n) {
        if(m == 0) {
            for(int i = 0; i < n; i++) {
                nums1[i] = nums2[i];
            }
        }
        int left = m - 1, right = n - 1;
        int index = n + m - 1;
        while(left >= 0 && right >= 0) {
            if(nums1[left] >= nums2[right]) {
                nums1[index--] = nums1[left--];
            }else {
                nums1[index--] = nums2[right--];
            }
        }
        if(right != -1) {
            while (right >= 0) {
                nums1[index--] = nums2[right--];
            }
        }
    }
}
