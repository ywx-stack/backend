# 剑指offer
本笔记用于记录三刷剑指offer，重点为实现自己对于题目的理解和解法。

[TOC]

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

- 时间复杂度`appendTail()`函数为 *O*(1) ；`deleteHead()` 函数在 *N* 次队首元素删除操作中总共需完成 *N* 个元素的倒序。
- 空间复杂度*O*(N):最差情况保存N个元素。

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

- 时间复杂度*O(1)*
- 空间复杂度*O(N)*

**<栈与队列>小总结：今天的两题都是对两个stack进行操作，一个用于接收新数据，另一个用于辅助完成对应的功能。很奇妙的思路。**

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

- 时间复杂度*O*(N)
- 空间复杂度*O*(N)

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

- 时间复杂度*O*(N)
- 空间复杂度*O*(N)

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

- 时间复杂度*O*(N) ：遍历一遍
- 空间复杂度*O*(1)：指针常数的额外空间

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

- 时间复杂度*O*(N)：一轮遍历。
- 空间复杂度*O*(N)：遍历链表的递归深度为N，使用N的额外空间。

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

- 时间复杂度*O*(N)： 两列遍历
- 空间复杂度*O*(N) ：哈希表`dic`使用线性大小的额外空间

```PYTHON
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

- 时间复杂度*O*(N) ：三轮遍历链表。
- 空间复杂度*O*(1) ：节点引用变量使用常数大小的额外空间。

**<链表（简单）>总结：**

**3道链表题，注意前两天用递归的时候，是在回溯的时候做需要的操作。**

**链表题的关键是对指针的理解，尤其涉及多个next的时候，画图更清晰。**

------

## 05.替换空格

感觉主要是考察c++语法的原地替换，但是python黑魔法。

```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        res =[]
        for c in s :
            if c == " ": res.append("%20")
            else : res.append(c)
        return ''.join(res)
        #join这个方法，将可迭代的数据类型，转为字符串或者bytes，没错可以转为bytes类型。注意这个可迭代的数据中的元素必须是相同类型的。
```

- 时间复杂度*O*(N) ：遍历链表。
- 空间复杂度*O*(N) 

```c++
// c++原地修改
class Solution {
public:
    string replaceSpace(string s) {
        int count = 0, len = s.size();
        // 统计空格数量
        for (char c : s) {
            if (c == ' ') count++;
        }
        // 修改 s 长度
        s.resize(len + 2 * count);
        // 倒序遍历修改
        for(int i = len - 1, j = s.size() - 1; i < j; i--, j--) {
            if (s[i] != ' ')
                s[j] = s[i];
            else {
                s[j - 2] = '%';
                s[j - 1] = '2';
                s[j] = '0';
                j -= 2;
            }
        }
        return s;
    }
};

作者：Krahets
链接：https://leetcode.cn/leetbook/read/illustration-of-algorithm/50c26h/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

- 时间复杂度*O*(N) ：遍历链表。
- 空间复杂度*O*(1) 

## 58-II.左旋转字符串

方法1：字符串切片

```python
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        return s[n:] + s[:n]
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*N*) 

方法2：列表遍历拼接

```python
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
		res = []
        for i in range(n,len(s)):
            res.append(s[i])
        for i in range(n):
            res.append(s[i])
        return ''.join(res)
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*N*) 

方法3：字符串遍历拼接

```python
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
		res = ''
        for i in range(n,len(s)):
            res += s[i]
        for i in range(n):
            res += s[i]
        return res
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*N*) 

方法4：三次翻转（C++）

```c++
class Solution {
public:
    string reverseLeftWords(string s, int n) {
        reverseString(s, 0, n - 1);
        reverseString(s, n, s.size() - 1);
        reverseString(s, 0, s.size() - 1);
        return s;
    }
private:
    void reverseString(string& s, int i, int j) {
        while(i < j) swap(s[i++], s[j--]);
    }
};

作者：Krahets
链接：https://leetcode.cn/leetbook/read/illustration-of-algorithm/58eckc/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*1*) 

此题的关键，也是对字符串做原地替换，但是这个的前提是C++，python的黑魔法不支持这些操作。

**<字符串（简单）>其实用python很简单，但是少了c++的考点，可以稍微补充一下c++的做法。**
