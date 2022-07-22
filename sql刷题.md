[TOC]

# 力扣专项：SQL入门

1.选择

2.排序 & 修改

3.字符串处理函数/正则

4.组合查询 & 指定选取

5.合并

# 595.大的国家

第一种解法：`or`关键字

```sql
SELECT 
	name,population,area
FROM
	world
WHERE
	area >=3000000 or population >= 25000000
```

第二种解法：`UNION`关键字

 union：会对两个结果集进行`并集`操作，不包括重复行，同时进行默认规则的排序。

```sql
SELECT 
	name,population,area
FROM 
	world
WHERE
	area >= 3000000
UNION
SELECT 
	name,population,area
FROM 
	world
WHERE
	population >= 25000000
```

因为用了索引，比第一种方法更快。

`or`和`UNION`的比较:

`对于单列来说，用or是没有任何问题的，但是or涉及到多个列的时候，每次select只能选取一个index，如果选择了area，population就需要进行table-scan，即全部扫描一遍，但是使用union就可以解决这个问题，分别使用area和population上面的index进行查询。 但是这里还会有一个问题就是，UNION会对结果进行排序去重，可能会降低一些performance(这有可能是方法一比方法二快的原因），所以最佳的选择应该是两种方法都进行尝试比较。`

# 584.寻找用户推荐人

字段为空的判断：`is null`  

字段不为空的判断：`is not null`

```sql
SELECT
	name
FROM
	customer
WHERE
	referee_id != 2 or referee_if is null
```

不等于还有一种写法：`<>`

# 183.从不订购的客户

`子查询` 和 `not in`的用法

```sql
SELECT
	name as customers
FROM
	Customers
WHERE
	customers.id not in
	(select customerid from orders)
```

------

# 1873.计算特殊奖金

`case when ... then ... else... end`

```sql
case 原表中的字段
when 原来的值1 then 新的值1
when 原来的值2 then 新的值2
when 原来的值3 then 新的值3
when 原来的值4 then 新的值4
else ''
end as 新表中的字段
```

`mod()`：余数

`rlike '^M'`：正则表达式，也称 模糊匹配，只要字段的值中存在要查找的部分 就会被选择出来。而`like`是全字段匹配。`regexp`与`rlike`相同。

`order by`：排序，默认升序，降序需要加`desc`

```sql
select
	employee_id,
case 
	when mod(employee_id,2) = 1 and name not rlike '^M' then salary
	else 0
end as bonus
from
	employees
order by employee_id;
```

# 627.变更性别

`update 表名 set x=?`更新表

同样涉及到`case when ...then...else...end`

```sql
update
	salary
set 
	sex =
	case
	when sex = 'm' then 'f'
	else 'm'
	end
```

# 196.删除重复的电子邮箱

`delete`注意判断条件的精妙

```sql
delete
	p1
from
	person p1,
	person p2
where
	p1.email = p2.email and p1.id>p2.id
```

# 1667.修复表中的名字

`concat()`:拼接

`uppper(str)`:变为大写

`lower(str)`：变为小写

`left(str,length)`:从左侧开始截取字符串

`substring(str,begin,end)`：截取字符串，end不写默认为空

```sql
select user_id,
	concat(upper(left(name,1)),lower(substring(name,2))) as name
from users
order by user_id
```

# 1484.按日期分组销售产品

`count()`：计数

`discinct`:不同的

`group_concat()`:

```
# 将分组中column1这一列对应的多行的值按照column2 升序或者降序进行连接，其中分隔符为seq
# 如果用到了DISTINCT，将表示将不重复的column1按照column2升序或者降序连接
# 如果没有指定SEPARATOR的话，也就是说没有写，那么就会默认以 ','分隔
GROUP_CONCAT([DISTINCT] column1 [ORDER BY column2 ASC\DESC] [SEPARATOR seq]);
```

```sql
select sell_date,
	count(distinct product) num_sold,
	group_concat(distinct product) products
from
	activities
group by sell_date
order by sell_date
```

# 1527.患某种疾病的患者

`rlike`更具体的用法

```sql
1、模糊查询字段中包含某关键字的信息。

如：查询所有包含“希望”的信息：select * from student where name rlike '希望'

2、模糊查询某字段中不包含某关键字信息。

如：查询所有包含“希望”的信息：select * from student where name not rlike '希望'

3、模糊查询字段中以某关键字开头的信息。

如：查询所有以“大”开头的信息：select * from student where name not rlike '^大'

4、模糊查询字段中以某关键字结尾的信息。

如：查询所有以“大”结尾的信息：select * from student where name not rlike '大$'

5、模糊匹配或关系，又称分支条件。

如：查询出字段中包含“幸福，幸运，幸好，幸亏”的信息：

select * from student where name  rlike '幸福|幸运|幸好|幸亏'

注意正则表达式或关系的表达方式为 |
————————————————
版权声明：本文为CSDN博主「GaoYan1024」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/GaoYan1024/article/details/122172278
```

```sql
SELECT * FROM PATIENTS
WHERE CONDITIONS REGEXP '^DIAB1|\\sDIAB1'
```

'\s'表示空格，MySQL 在插入数据库的时候，会自动去除转义字符也就是反斜杠“\”。

# 1965.丢失信息的雇员

`union all `：不带自动去重的联结两表

`having`的用法：`查询语句 + group by + having +聚合函数统计`，常见的聚合函数有`sum(),avg(),count()`。

```sql
select employee_id
from (
    select employee_id from Employees
    union all
    select employee_id from Salaries
) as t
group by employee_id
having count(employee_id) = 1
order by employee_id
```

# 1795.每个产品在不同商店的价格

横表转竖表

```sql
select product_id,'store1' as store, store1 as price
from Products where store1 is not null
union all 
select product_id,'store2' as store, store2 as price
from Products where store2 is not null
union all 
select product_id,'store3' as store, store3 as price
from Products where store3 is not null
```

# 608.树节点

颇有结合算法和sql的奇妙之处

```sql
select id
case 
	when tree.id = (select atree.id from tree atree where p_id is null) then 'Root'
	when tree.id in (select atree.p_id from tree atree) then 'Inner'
	else 'Leaf'
end as type
from tree
order by id
```

另一方法：使用`IF`函数，来避免复杂的流控制语句。

```
IF( expr1 , expr2 , expr3 )
expr1 的值为 TRUE，则返回值为 expr2
expr1 的值为FALSE，则返回值为 expr3
```

```sql
SELECT
    atree.id,
    IF(ISNULL(atree.p_id),
        'Root',
        IF(atree.id IN (SELECT p_id FROM tree), 'Inner','Leaf')) Type
FROM
    tree atree
ORDER BY atree.id

作者：LeetCode
链接：https://leetcode.cn/problems/tree-node/solution/shu-jie-dian-by-leetcode/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

# 176.第二高的薪水

`ifnull(expr1,alt_value)`与上述`if` 类似；

`limit n` 子句表示查询结果返回前n条数据

`offset n` 表示跳过x条语句

`limit y offset x `分句表示查询结果跳过 x 条数据，读取前 y 条数据

```sql
select 
	ifnull((select distinct salary from employee 
            order by salary desc
            limit 1 offset 1),null) 
            as secondhightestsalary
```

# 175.组合两个表

![img](https://pic.leetcode-cn.com/ad3df1c4ecc7d2dbe85f92cdde8ec9a731fdd20dc4c5629ecb372b21de26c682-1.jpg)

```sql
select firstname,lastname,city,state
from person left join address
on person.personid = address.personid
```

# 1581.进店却未进行过交易的顾客

上图的左下的情况。

```sql
SELECT 
    customer_id, COUNT(customer_id) count_no_trans
FROM 
    visits v
LEFT JOIN 
    transactions t ON v.visit_id = t.visit_id
WHERE amount IS NULL
GROUP BY customer_id;
```

# 1148.文章浏览I

很水

```sql
select distinct author_id as id
from views
where author_id = viewer_id
order by id
```

------

# 197.上升的温度

趁着把《MySQL必知必会》第15章和第16章过一遍，讲了联结和高级联结；

**内部联结or等值联结：**

下面两种写法是一致的

```sql
select a,b,c
from t1 inner join t2
on t1.id=t2.id
```

```sql
select a,b,c
from t1,t2
where t1.id = t2.id
```

推荐使用`inner join`的写法。

如果没有后面的条件，就是返回 笛卡儿积（第一个表的行与第二个表的每一行配对）,也被称为`cross join`。

**自联结：**(自己和自己联结，用于查找符合某条件的其它数据)

```sql
select t1.a,t1.b
from table1 as t1,table2 as t2
where t1.id = t2.id and t2.id='aaa'
```

**外联结**：可以参考上上图

**本题还涉及对时间做差，可使用**

`datediff(d1,d2)`：返回天数差

```sql
select w1.id
from weather w1 inner join weather w2
on datediff(w1.recordDate , w2.recordDate) = 1
where w1.Temperature >  w2.Temperature 
```

# 607.销售员

没什么新知识点。

```sql
SELECT
    s.name
FROM
    salesperson s
WHERE
    s.sales_id NOT IN (SELECT
            o.sales_id
        FROM
            orders o
                LEFT JOIN
            company c ON o.com_id = c.com_id
        WHERE
            c.name = 'RED')
;

作者：LeetCode
链接：https://leetcode.cn/problems/sales-person/solution/xiao-shou-yuan-by-leetcode/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

# 1141.查询近30天活跃用户数

可以用`between and`

```sql
select activity_date as day,count(distinct user_id) as active_users
from activity
where datediff('2019-07-27',activity_date) < 30 and datediff('2019-07-27',activity_date) > 0
group by activity_date
```

# 1693.每天的领导和合伙人

```sql
select date_id,make_name,count(distinct lead_id) as unique_leads,count(distinct partner_id) as unique_partners
from dailysales
group by date_id , make_name
```

# 1729.求关注者的数量

```sql
select user_id,count(distinct follower_id) as followers_count
from followers
group by user_id
order by user_id
```

# 586.订单最多的客户

我的答案

```sql
select customer_number
from (select customer_number, count(customer_number) as n
from orders
group by customer_number
order by n desc limit 1 ) a
```

官方题解

```sql
SELECT
    customer_number
FROM
    orders
GROUP BY customer_number
ORDER BY COUNT(*) DESC
LIMIT 1
```

# 511.游戏玩法分析I

`min（）`

```sql
select player_id,min(event_date) as first_login
from activity
group by player_id
```

# 1890.2020年最后一次登录

```sql
select user_id,max(time_stamp) as last_stamp
from logins
where time_stamp between '2020-01-01 00:00:00' and '2020-12-31 23:59:59'
group by user_id
```

网上题解，注意`year()`

```sql
SELECT user_id, max(time_stamp) last_stamp      #求最大的日期用max，但是如何限定是2020年呢？
FROM Logins
WHERE year(time_stamp) = '2020'                      #看这！！！！！！！用year函数增加条件为2020年
GROUP BY user_id;                                              #这个好理解就是分个组
```

# 1741.查找每个员工花费的总时间

```sql
select event_day as day,emp_id,sum(out_time-in_time) as total_time
from employees
group by emp_id,event_day
```

