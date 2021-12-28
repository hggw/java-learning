package LeetCodeShuangZhiZhen;

/**
 * 167. 两数之和 II - 输入有序数组
 */
public class TwoSumII {
    public int[] twoSum(int[] numbers, int target) {
        int length = numbers.length;
        int left = 0, right = length - 1;
        while(left < right) {
            int tmp = numbers[left] + numbers[right];
            if(tmp == target) {
                break;
            } else if (tmp > target){
                right--;
            } else {
                left++;
            }
        }
        return new int[]{left+1,right+1};
    }
}
