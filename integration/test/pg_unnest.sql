--本测试SQL使用tpcds数据表

--解嵌套关联子查询testcase1:
--测试优化器是否能够解嵌套带有标量聚合子查询op常量的查询，这是一个带有LIMIT的版本
--下面第一个语句是原始查询，在解嵌套之后应该在查询树上和第二个查询有着类似的结构
SELECT c.c_last_name FROM customer c WHERE (SELECT SUM(ss_sales_price) FROM store_sales ss WHERE ss.ss_customer_sk = c.c_customer_sk) > 5000 LIMIT 100;
SELECT c.c_last_name FROM customer c JOIN (SELECT ss_customer_sk, SUM(ss_sales_price) AS total_sales FROM store_sales GROUP BY ss_customer_sk HAVING SUM(ss_sales_price) > 5000 LIMIT 100) ss ON c.c_customer_sk = ss.ss_customer_sk;

--解嵌套关联子查询testcase2:
--测试优化器是否能够解嵌套带有GROUP BY且依赖值依赖层数更深的标量聚合子查询op常量的查询
SELECT i.i_product_name FROM item i WHERE (SELECT SUM(total_price) FROM (SELECT ss_item_sk, SUM(ss_sales_price) AS total_price FROM store_sales ss WHERE ss.ss_item_sk = i.i_item_sk GROUP BY ss.ss_item_sk)) > 5000 LIMIT 10;
SELECT i.i_product_name FROM item i JOIN (SELECT ss_item_sk, SUM(ss_sales_price) AS total_price FROM store_sales ss GROUP BY ss_item_sk HAVING SUM(ss_sales_price) > 5000 LIMIT 10) ss ON i.i_item_sk = ss.ss_item_sk;

--解嵌套关联子查询testcase3:
--测试一种更一般的情况:
SELECT * FROM item i WHERE (SELECT SUM(t.ss_sales_price) FROM (SELECT * FROM store_sales ss,customer c WHERE ss.ss_item_sk=i.i_item_sk AND ss.ss_customer_sk=c.c_customer_sk ) t) > 5000 LIMIT 10;
SELECT * FROM item i JOIN (SELECT ss_item_sk, SUM(ss_sales_price) AS total_sales FROM store_sales ss GROUP BY ss_item_sk HAVING SUM(ss_sales_price) > 5000 LIMIT 10) ss ON i.i_item_sk = ss.ss_item_sk;

