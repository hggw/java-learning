package TestDemo.Test03;

public class Sort {

    public static void main(String[] args) {
        int[] nums = new int[]{1,4,2,3,5,9,8,7,6};
        int[] temp = new int[nums.length];
        //choiceSort(nums);
        //quickSort(nums,0,nums.length - 1);
        mergeSort(nums,0,nums.length,temp);


    }

    public static void choiceSort(int[] nums) {
        int length = nums.length;
        for(int i = 0; i < length; i++) {
            int minPos = i;
            for(int j = i + 1; j < length; j++) {
                if(nums[j] < nums[minPos]) {
                    minPos = j;
                }
            }
            if(minPos != i) {
                nums[i] = (nums[i]) ^ (nums[minPos]);
                nums[minPos] = nums[i] ^ nums[minPos];
                nums[i] = nums[i] ^ nums[minPos];
            }
        }
        for(int num : nums) {
            System.out.println(num);
        }
    }
    public static void quickSort(int[] nums, int left,int right) {
        if(left + 1 >= right) {
            return;
        }
        int first = left, last = right - 1;
        int key = nums[first];
        while(first < last) {
            while(first < last && nums[last] >= key) {
                last--;
            }
            nums[first] = nums[last];
            while (first < last && nums[first] <= key) {
                first++;
            }
            nums[last] = nums[first];
        }
        nums[first] = key;
        quickSort(nums,0,first);
        quickSort(nums,first+ 1,right);
    }

    public static void mergeSort(int[] nums,int left , int right ,int[] temp) {
        if(left + 1 >= right) {
            return;
        }
        int mid = (left + right)/2;
        mergeSort(nums,left,mid,temp);
        mergeSort(nums,mid,right, temp);

        int p = left, q = mid,i = left;
        while(p < mid || q < right) {
            if(q >= right || (p < mid && nums[p] <= nums[q])) {
                temp[i++] = nums[p++];
            } else {
                temp[i++] = nums[q++];
            }
        }
        for(int j = left; j < right; j++) {
            nums[j] = temp[j];
        }
    }
}
