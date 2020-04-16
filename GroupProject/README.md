# Group Member

>* Zhang Yijia,1801212815,YijiaZhang1996:[GitHub link](https://github.com/YijiaZhang1996)
>* Deng Ying,1801212782,dy0703:[GitHub link](https://github.com/dy0703)
>* Chen Zhuo,1901212461,Igloo7:[GitHub link](https://github.com/Igloo7/Igloo)
>* Wang Congyuan,1901212637,cy-wang15:[GitHub link](https://github.com/cy-wang15)

# Research Topic
O2O Coupon Consumption Prediction Based on Past Consumer Behavior 

# Research Background

In recent years, O2O (Online to Offline) consumption has gradually entered people's life. According to incomplete statistics, the O2O industry is associated with hundreds of millions of consumers, and all kinds of apps record more than 10 billion user behavior and location records every day. Activating old users with coupons or attracting new customers to shop is an important marketing method of O2O. However, randomly placed coupons cause meaningless interference to most users. For merchants, spamming coupons can reduce brand reputation and make it difficult to estimate marketing costs.

# Research Purpose

This project aims to predict users' usage within 15 days after receiving coupons by analyzing and modeling their past offline consumption behaviors, so as to realize personalized coupon delivery. Make consumer gets real benefit, also give businessman stronger sale ability at the same time.

# Data Source
This project and data are derived from Tianchi Big Data Contest. Tianchi big data contest is a big data modeling contest held by Alibaba group, similar to Kaggle.
Data Source: [link](https://tianchi.aliyun.com/competition/entrance/231593/information)


# Data Description
There are three tables in this project. They are:

> * ccf_offline_stage1_test_revised.csv
> * ccf_offline_stage1_train.csv
> * Submit File

Description of each table (including features and explanations) are listed as follows:

Table1 Users' Offline Consumption and Coupon Collection Behavior
| Feature|Description|
| -------- | :----:  |
| User_id  |  User ID       |
|Merchant_id|	Merchant ID
|Coupon_id|	"Null" means there is no coupon consumption, and the Discount_rate and Date_received fields are meaningless in this situation.
|Discount_rate|	Represents the discount rate; x: y means x minus y. Unit: Yuan
|Distance|	The location where the user frequently goes to is 500*x meters away from the nearest store of the merchant (if it is a chain store, the nearest store is taken); null means no such information, and 0 means the distance is less than 500 meters, 10 means the distance is more than 5 kilometers.
|Date_received|	The date of receiving the coupon.
|Date|If (date = null) & (coupon-id != null), the record indicates that the coupon is collected but not used, that is, negative sample; if (date! = null) & (coupon-id = null),it indicates the ordinary consumption date; if (date != null) & (coupon-id != null),it indicates the coupon consumption date, that is, positive sample.|


Table2 Users O2O Offline Coupon Usage Prediction Sample
| Feature|Description|
| -------- | :----:  |
| User_id  |  User ID       |
|Merchant_id|	Merchant ID
|Coupon_id|	"Null" means there is no coupon consumption, and the Discount_rate and Date_received fields are meaningless in this situation.
|Discount_rate|	Represents the discount rate; x: y means x minus y. Unit: Yuan
|Distance|	The location where the user frequently goes to is 500*x meters away from the nearest store of the merchant (if it is a chain store, the nearest store is taken); null means no such information, and 0 means the distance is less than 500 meters, 10 means the distance is more than 5 kilometers.|
|Date_received|	The date of receiving the coupon.|
|Date|If (date = null) & (coupon-id != null), the record indicates that the coupon is collected but not used, that is, negative sample; if (date! = null) & (coupon-id = null),it indicates the ordinary consumption date; if (date != null) & (coupon-id != null),it indicates the coupon consumption date, that is, positive sample.|

Table3 Submit File
| Feature|Description|
| -------- | :----:  |
| User_id  |  User ID       |
|Coupon_id |“Null” means there is no coupon consumption, and the Discount_rate and Date_received fields are meaningless in this situation.|
|Date_received|The date of receiving the coupon.|
|Probability|The Probability that we need to predict and used for scoring.|

This table is used to save prediction result(i.e. Probability) and result submission. After submitting it to Tianchi Scoring system, you can get score for your model.
