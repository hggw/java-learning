package LeetCodeTanXin.easy;

/**
 * 605. 种花问题
 * @author HGW
 */
public class CanPlaceFlowers {
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int length = flowerbed.length;
        int count = 0;
        if(length == 1 && flowerbed[0] == 0) {
            flowerbed[0] = 1;
            count++;
        }
        if(length == 2 && flowerbed[0] + flowerbed[1]== 0) {
            flowerbed[0] = 1;
            count++;
        }
        for(int i = 1; i < length - 1; i++) {
            if( i-1 == 0 && flowerbed[i-1] == 0 && flowerbed[i] == 0) {
                flowerbed[i-1] = 1;
                count++;
            }
            if(i+2 == length && flowerbed[i] == 0 && flowerbed[i+1] == 0) {
                flowerbed[i+1] = 1;
                count++;
            }
            if(flowerbed[i-1] == 0 && flowerbed[i] == 0 && flowerbed[i+1] == 0) {
                flowerbed[i] = 1;
                count++;
            }
        }
        return count >= n;
    }
}
