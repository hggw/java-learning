package LeetCodeTanXin.easy;

/**
 * 122. 买卖股票的最佳时机 II
 * @author HGW
 */
public class BestTimeBuySellStockII {

    public int maxProfit(int[] prices) {
        int length = prices.length;
        int profit = 0;
        for(int i = 0; i < length - 1; i++) {
            if(prices[i] < prices[i+1]) {
                profit += prices[i+1] - prices[i];
            }
        }
        return profit;
    }
}
