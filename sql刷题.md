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

`rlike '^M'`：正则表达式，也称 模糊匹配，只要字段的值中存在要查找的部分 就会被选择出来。而`like`是全字段匹配。

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

