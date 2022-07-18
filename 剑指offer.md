# 剑指offer
本笔记用于记录三刷剑指offer，重点为实现自己对于题目的理解和解法。

## 9.用两个栈实现队列

![image-20220717140147608](picture/image-20220717140147608.png)

```python
class CQueue:

    def __init__(self):
        self.A ,self.B = [],[]

    def appendTail(self, value: int) -> None:
        self.A.append(value)

    def deleteHead(self) -> int:
        if not self.B:#为空时
            while self.A: #从A拿过来
                self.B.append(self.A.pop())
            if self.B:
                return self.B.pop()
            else:
                return -1
        else :
            return self.B.pop()


# Your CQueue object will be instantiated and called as such:
# obj = CQueue()
# obj.appendTail(value)
# param_2 = obj.deleteHead()
```

## 30.包含min函数的栈

用两个stack，分别功能如下：

![image-20220717135358548](picture/image-20220717135358548.png)

```python
class MinStack:
    def __init__(self): #注意后续都要加self
        self.A, self.B = [],[]

    def push(self,x:int) -> None:
        self.A.append(x)
        if not self.B or self.B[-1] >=x: #注意是大于等于
            self.B.append(x)
      
    def pop(self) -> None:
        if self.A.pop() == self.B[-1]:
            self.B.pop()
        
    def top(self) -> int :
    	return self.A[-1]
    
    def min(self) -> int :
        return self.B[-1]
```

<栈与队列>小总结：今天的两题都是对两个stack进行操作，一个用于接收新数据，另一个用于辅助完成对应的功能。很奇妙的思路。

------



## 6.从尾到头打印链表

```python
#递归
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        def recur(head):
            if not head:return
            recur(head.next)
            res.append(head.val)
     	res = []
        recur(head)
        return res
```

```python
#辅助栈
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        res = []
        while head!=None :
            res.append(head.val)
            head=head.next
        return res[::-1]
```

## 24.反转列表

<img src="picture/image-20220718124658595.png" alt="image-20220718124658595" style="zoom:50%;" />

```python
#迭代（双指针）
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cur = head
        pre = None
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre
```

递归遍历链表，当越过尾结点后终止递归，回溯时修改各节点的`next`引用指向（这是递归的精髓）

```python
#递归
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        def recur(cur,pre):
            if not cur:return pre #终止条件
            res = recur(cur.next,cur)#递归后继节点
            cur.next = pre #修改节点引用指向
            return res # 返回反转链表的头结点
        return recur(head,None)
```

这个递归代码很巧妙的一步是`res`,一直都是`最后一个节点的位置`！

## 35.复杂链表的复制

当考虑直接复制的时候，由于`.random`无法确定，因此无法直接遍历一遍复制得到结果。

两种方法：`哈希表`和 `拼接+拆分`

```python
# 哈希表
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return None
        cur = head
        dic = {}
        # 构建 原链表->新链表 的字典
        while cur:
            dic[cur] = Node(cur.val)
            cur = cur.next
        cur = head
        # 构建 新节点的next和random指向
        while cur:
            dic[cur].next = dic.get(cur.next)
            dic[cur].random = dic.get(cur.random)
            cur = cur.next
        return dic[head]
```

```python
# 拼接+拆分
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head : return None
        cur = head
        # 1.复制链表 变为node1->new node1 -> node2 -> new node2 ...
        while cur:
            node = Node(cur.val)
            node.next = cur.next
            cur.next = node
            cur = node.next
        # 2.给new链表指向相应的指针
        cur = head
        while cur:
            if cur.random:
                cur.next.random = cur.random.next
            cur = cur.next.next
        # 3.拆分（不能一步到位，否则两个链表还是有关联的）
        res = cur = head.next
        pre = head
        while cur.next:
            pre.next = pre.next.next
            cur.next = cur.next.next
            pre = pre.next
            cur = cur.next
        pre.next = None
        return res
```

着重注意拆分的时候的代码，顺序是有问题的，建议画图分析。

<链表（简单）>总结：

- 3道链表题，注意前两天用递归的时候，是在回溯的时候做需要的操作。
- 链表题的关键是对指针的理解，尤其涉及多个next的时候，画图更清晰。
