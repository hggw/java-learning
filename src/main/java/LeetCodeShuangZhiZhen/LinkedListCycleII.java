package LeetCodeShuangZhiZhen;

import LeetCodeExcise.ListNode;

/**
 * 142. 环形链表 II
 * @author HGW
 */
public class LinkedListCycleII {
    public ListNode detectCycle(ListNode head) {
        ListNode front = head, back = head;
        while(front != null && back != null ) {
            front = front.next;
            if(back.next != null) {
                back = back.next.next;
            } else {
                return null;
            }
            if(front == back) {
                ListNode res = head;
                while (res != front) {
                    res = res.next;
                    front = front.next;
                }
                return res;
            }
        }
        return null;
    }
}
