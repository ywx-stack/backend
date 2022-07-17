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

小总结：今天的两题都是对两个stack进行操作，一个用于接收新数据，另一个用于辅助完成对应的功能。很奇妙的思路。
