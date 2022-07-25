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

------

## 03.数组中重复的数字

思路1：自己想的最直观的方法

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        # 排序，遍历
        nums.sort()
        for i in range(1,len(nums)):
            if nums[i-1] == nums[i]:
                return nums[i]
        return nums[0]
```

- 时间复杂度*O*(*N*+logN):遍历+排序所需的时间
- 空间复杂度*O*(*1*) ：不引入额外空间

思路2：借助哈希/set

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        hash = {}
        for num in nums:
            if num in hash:
                return num
            hash[num] = num
        return 0
```

- 时间复杂度*O*(*N*):遍历一遍
- 空间复杂度*O*(*N*) 

因为只需要存一个值，可以用`set`

```python
class Solution:
    def findRepeatNumber(self, nums: [int]) -> int:
        dic = set()
        for num in nums:
            if num in dic: return num
            dic.add(num)
        return -1
```

- 时间复杂度*O*(*N*):遍历一遍
- 空间复杂度*O*(*N*) 

方法3：原地交换

一个萝卜一个坑，如果坑里有萝卜，肯定是重复的萝卜了

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            if nums[i] != i:
                if nums[nums[i]] == nums[i]:
                    return nums[i]
                nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        return nums[0]
```

- 时间复杂度*O*(*N*):遍历一遍
- 空间复杂度*O*(*1*) 

## 53-I.在排序数组中查找数字

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # 排好序的可以用二分查找
        l, r = 0, len(nums)-1
        while l <= r:
            mid = (l + r) // 2
            # 中间值 比 目标值 大 ，在左侧
            if nums[mid] > target:
                r = mid - 1
            elif nums[mid] < target: #在右侧
                l = mid + 1
            else:# 相等
                if nums[r] != target:
                    r -= 1
                elif nums[l] != target:
                    l += 1
                else:
                    break
        return r - l + 1
```

- 时间复杂度*O*(*logN*):二分
- 空间复杂度*O*(*1*) 

## 53-II.0~n-1中缺失的数字

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        # 二分
        l, r = 0, len(nums)-1
        while l <= r:
            mid = (l+r)//2
            if nums[mid] > mid:
                r = mid - 1
            elif nums[mid] == mid:
                l = mid + 1
        return l
```

- 时间复杂度*O*(*logN*):二分
- 空间复杂度*O*(*1*) 

**<查找算法（简单）>注意排序数组中用二分法查找最快，默认下标当索引，重点是利用到数组的特性。**

## 04.二维数组中的查找

关键在于数组特性，从左到右递增，从上到下也递增

```python
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix : return False
        n = len(matrix)
        m = len(matrix[0])
        i, j = n-1, 0
        while i >= 0 and j < m:
            if matrix[i][j] > target:
                i -= 1
            elif matrix[i][j] < target:
                j += 1
            else:
                return True
        return False
```

- 时间复杂度*O*(*N*+*M*):遍历一遍
- 空间复杂度*O*(*1*) 

## 11.旋转数组的最小数字

这道题的分析值得多看两遍，还是很复杂的。尤其是对等号的判断。

看官方题解的视频，可以理解成 二分和暴力法的结合！

```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        # 二分 + 暴力
        l, r = 0, len(numbers)-1
        while l < r:
            mid = (l+r)//2
            if numbers[mid] > numbers[r]:
                l = mid + 1
            elif numbers[mid] < numbers[r]:
                r = mid 
            else:
                r -= 1 #暴力
        return numbers[l]
```

- 时间复杂度*O*(*logN*)，但是在数组全部相等的情况下是O(N)
- 空间复杂度*O*(*1*) 

## 50.第一个只出现一次的字符

暴力法了话是O（N^2)，借助常规的空间换时间的思路，引入哈希表：

```python
class Solution:
    def firstUniqChar(self, s: str) -> str:
        dic = {}
        for c in s:
            if c in dic:
                dic[c] += 1
            else:
                dic[c] = 1
        for c in s:
            if dic[c] == 1:
                return c
        return ' '
```

- 时间复杂度*O*(*N*):遍历2遍
- 空间复杂度*O*(*1*) :虽然引入了哈希，但是只有26个字母*O*(*26*) =*O*(*1*)

因为python3.6之后，默认字典是有序的，因此可以写成下面：

```python
class Solution:
    def firstUniqChar(self, s: str) -> str:
        dic = {}
        for c in s:
            dic[c] = not c in dic
        for k, v in dic.items():
            if v: return k
        return ' '
```

相对第一种方法，仅遍历了一遍。

**<查找算法（中等）>，今天的三道题，重点还是分析题目的特性；不过不建议直接背题目的最优解，要自行学习分析思路，第二道旋转数组的最小数字很值得学习，一步一步的分析。**

------

## 32-I.从上到下打印二叉树

二叉树的广度优先搜索（`BFS`）,需要借助队列

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root:return []
        deque = collections.deque()
        res = []
        deque.append(root)
        while deque:
            node = deque.popleft()
            res.append(node.val)
            if node.left:deque.append(node.left)
            if node.right: deque.append(node.right)
        return res
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*N*) 

## 32-II.从上到下打印二叉树II

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:return []
        deque = collections.deque()
        res = []
        deque.append(root)
        while deque:
            tmp = []
            n = len(deque)
            for i in range(n):
                node = deque.popleft()
                tmp.append(node.val)
                if node.left:deque.append(node.left)
                if node.right:deque.append(node.right)
            res.append(tmp)
        return res
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*N*) 

## 32-III.从上到下打印二叉树III

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:return []
        deque = collections.deque()
        res = []
        deque.append(root)
        flag = 0
        while deque:
            tmp = collections.deque()
            for _ in range(len(deque)):
                node = deque.popleft()
                if flag%2 == 0:
                    tmp.append(node.val)
                else:
                    tmp.appendleft(node.val)
                if node.left:deque.append(node.left)
                if node.right:deque.append(node.right)
            flag += 1
            res.append(list(tmp))
        return res
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*N*) 

**<搜索与回溯算法（简单）>主要是二叉树的层序遍历，记住要借助双端队列就可以！**

## 26.树的子结构

两个递归；

一个递归是A,一个递归是进行A、B相等判断；

然后分析每一个的终止条件和返回条件；

![image-20220723193819530](picture/image-20220723193819530.png)

```python
class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        if not A or not B:return False
        def recur(A,B):
            if not B:return True
            if not A or A.val !=B.val :return False
            return recur(A.left,B.left) and recur(A.right,B.right)
        return recur(A,B) or self.isSubStructure(A.left,B) or self.isSubStructure(A.right,B)
```

- 时间复杂度*O*(*MN*)
- 空间复杂度*O*(*N*) 

## 27.二叉树的镜像

递归：

```python
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if not root:return
        tmp = root.left
        root.left = root.right
        root.right = tmp
        self.mirrorTree(root.left)
        self.mirrorTree(root.right)
        return root
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*N*) 

辅助栈：

```python
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        #辅助栈仅用于遍历每个节点
        if not root : return root
        res = []
        res.append(root)
        while res:
            node = res.pop()
            if node.left : res.append(node.left)
            if node.right : res.append(node.right)
            node.left ,node.right = node.right, node.left
        return root
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*N*) 

## 28.对称的二叉树

很巧妙的思路

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        # 先遍历，再验证
        def recur(L,R):
            if not L and not R:return True
            if not L or not R or L.val != R.val:return False
            return recur(L.left,R.right) and recur(L.right,R.left)
        return not root or recur(root.left,root.right)
```

**<搜索与回溯算法（简单）>其实和递归的思路很一致，以某种方式遍历，遍历的时候增加条件判断。**

## 10-I.斐波那契数列

```python
class Solution:
    def fib(self, n: int) -> int:
        a,b=0,1
        for _ in range(n):
            a,b = b,(a+b)%1000000007
        return a
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*1*) 

## 10-II.青蛙跳台阶问题

```python
class Solution:
    def numWays(self, n: int) -> int:
        if n == 0: return 1
        if n == 1: return 1
        if n == 2: return 2
        a,b = 1,2
        for _ in range(n-2):
            a ,b = b ,(a+b)%1000000007
        return b
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*1*) 

## 63.股票的最大利润

![img](https://pic.leetcode-cn.com/1600880605-QGgqZW-Picture1.png)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:return 0
        m_min = prices[0]
        res = 0
        for i in prices:
            if i < m_min:
                m_min = i
            res = max(i-m_min,res)
        return res
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*1*) 

## 42.连续子数组的最大和

动态规划!

![img](https://pic.leetcode-cn.com/77d1aa6a444743d3c8606ac951cd7fc38faf68a62064fd2639df517cd666a4d0-Picture1.png)

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1,len(nums)):
            nums[i] += max(nums[i-1],0)
        return max(nums)
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*1*) 

## 47.礼物的最大价值

与上一题思路类似

```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        i,j = 0, 0
        for i in range(n):
            for j in range(m):
                if i == 0 and j==0:
                    continue
                elif i == 0:
                    grid[i][j] += grid[i][j-1]
                elif j == 0:
                    grid[i][j] += grid[i-1][j]
                else:
                    grid[i][j] += max(grid[i][j-1],grid[i-1][j])
        return grid[n-1][m-1]
```

- 时间复杂度*O*(*NM*)
- 空间复杂度*O*(*1*) 

## 46.把数字翻译成字符串

类似跳台阶，但是要增加判断条件;

<img src="https://pic.leetcode-cn.com/1603462412-iUcKzA-Picture1.png" alt="img" style="zoom:50%;" />

注意画图，画值分析，而不是一直在脑海中想。

```python
class Solution:
    def translateNum(self, num: int) -> int:
        a = 1
        b = 1
        y = num % 10
        while num>9:
            num //= 10
            x = num % 10
            if 9 < x*10 + y <26:
                c = a + b
            else:
                c = a
            a ,b = c, a
            y = x
        return a
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*1*) 

## 48.最长不含重复字符的子字符串

动态规划:

![img](https://pic.leetcode-cn.com/1599287290-mTdFye-Picture1.png)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic = {}
        res = tmp = 0
        for j in range(len(s)):
            i = dic.get(s[j],-1) # 获取索引 i
            dic[s[j]] = j # 更新哈希表
            tmp = tmp + 1 if tmp < j - i else j - i
            res = max(res,tmp)
        return res
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*1*) 

哈希表+双指针

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic = {}
        i = -1
        res = 0
        for j in range(len(s)):
            if s[j] in dic:
                i = max(dic[s[j]],i)
            dic[s[j]] = j
            res = max(res,j-i)
        return res
```

- 时间复杂度*O*(*N*)
- 空间复杂度*O*(*1*) 
