# SiameseBert

1、取mean pool的输出v1, v2；
2、取CLS为位置的输出v1, v2；
后面直接拼接v1*v2, |v1-v2|

还有一种是参考ESIM，在BERT后面加入交互，计算align后做交互；

# DeFormer
自己想的，后来发现已经发表在ACL2020上了，大家参考一下那篇paper；
