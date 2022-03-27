#!/usr/bin/env python
# -*- coding : utf-8 -*-
"""
--------------------------------------------
@Time ： 7/08/2021 8:47 pm
@Auth : Andy yang
@File : 词.py
@IDE : PyCharm
@Email : quecheny@student.unimelb.edu.au
---------------------------------------------
"""
# 导入库
import jieba.posseg
import jieba.analyse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# 正则包
# 自然语言处理包
import jieba.analyse
# html 包
import html
# 导入库
import pandas as pd
import xlrd
import re # 正则表达式库
import collections # 词频统计库
import numpy as np # numpy数据处理库
import jieba # 结巴分词
import wordcloud # 词云展示库
from PIL import Image # 图像处理库
import matplotlib.pyplot as plt # 图像展示库
import datetime
import sys,codecs
import jieba.posseg

# 读取文件
fn = open('/Users/19723/Desktop/tyty.txt', encoding='UTF-8') # 打开文件
string_data = fn.read() # 读出整个文件
fn.close() # 关闭文件

# 文本预处理
pattern = re.compile(u'\t|\n|\.|-|:|;|\)|\(|\?|"') # 定义正则表达式匹配模式
string_data = re.sub(pattern, '', string_data) # 将符合模式的字符去除

# 文本分词
seg_list_exact = jieba.cut(string_data, cut_all = False) # 精确模式分词
object_list = []
remove_words = [u'的', u'，',u'和', u'是', u'随着', u'对于',u'对',u'等',u'能',u'都',u'。',
u' ',u'、',u'中',u'在',u'了',u'通常',u'如果',u'我们',u'需要', u'财联社', u'日讯', u"'", u',', u'【', u'】'
, u'联社', u':', u'“', u'”', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8',
                u'9', u'（', u'）', u'《', u'》', u'、', u'\\', u';', u'并', u'至',
               u'称', u'于', u'要求', u'已', u'企业', u'行业', u'预计', u'产品', u'其', u'较',
               u'未来', u'2020', u'有', u'公司', u'表示', u'进行', u'：', u'·',
               u'/', u'据', u'相关', u'与', u'将', u'为', u'月', u'目前', u'日', u'后来', u'·', u'；', u'/',
            u'但', u'个', u'上', u'及', u'下', u'部分', u'后', u'还' ,u'一些' ,u'从' ,u'也' ,u'或' ,u'已经' ,
                u'这', u'就', u'元', u'其中', u'可', u'向', u'超', u'报', u'年', u'正在', u'该', u'只' ,u'财' ,u'还' ,u'前' ,u'内',
                u'时' ,u'以及' , u'例', u'可能' , u'近日', u'不'  ,u'内', u'自',u'被' ,u'均' ,u'㎡' ,u'型', u'要', u'均', u'以',
                u'报道', u'今日', u'累计', u'此前', u'做', u'要', u'而', u'亿美元', u'亿元', u'会', u'以来', u'涉及', u'成为',
                u'不再', u'到', u'收到', u'截止', u'正', u'没有', u'经', u'点', u'万亿', u'n', u'来', u'当前', u'消息', u'仍',
                u'10', u'11', u'12', u'13', u'14', u'15', u'16', u'17', u'18' ,u'19', u'万亿元',
                u'20' ,u'21' ,u'22' ,u'23', u'24', u'25', u'26', u'27', u'28', u'29', u'30', u'31',
                u'指出', u'更', u'天', u'占', u'通过', u'认为', u'发现', u'同时', u'认为', u'家', u'桶',
                u'—', u'次', u'且', u'因', u'本', u'同花顺', u'雪', u'球网', u'市场', u'同比', u'2021', '*'
               ]
# 自定义去除词库

for word in seg_list_exact: # 循环读出每个分词
    if word not in remove_words: # 如果不在去除词库中
        object_list.append(word) # 分词追加到列表

# 词频统计
word_counts = collections.Counter(object_list) # 对分词做词频统计
word_counts_top200 = word_counts.most_common(100000) # 获取前10最高频的词
# print(word_counts_top200) # 输出检查
abc = pd.DataFrame(word_counts_top200)
abc.columns = ["keywords", "counts"]
abc
list1 = []
list2 = ['采掘','化工','钢铁','有色金属','建筑材料','建筑装饰','电气设备','机械设备','国防军工','汽车','家用电器','纺织服装','轻工制造','商业贸易','农林牧渔','食品饮料','休闲服务','医药生物','公用事业',
         '交通运输','房地产','电子','计算机','传媒','通信','银行','非银','综合']
caijue = ['石油开采', '开采','采掘', '油气钻采', '其他采掘', '焦炭加工', '焦煤', '煤炭', '炼化']
huagong = ['石油化工','石油贸易' , '化学原料', '纯碱', '氯碱', '无机盐', '氮肥', '磷肥', '农药', '聚氨酯','玻纤','纤维','塑料','橡胶','轮胎','合成革','PTA']
gangtie = ['钢铁', '普钢', '特钢', '钢材', '铁矿石']
yousejinshu = ['工业金属','铝','铜','铅','锌','黄金','稀有金属','稀土','钨','锂','锂电','锂电池','磁性材料','非金属新材料','磁铁','锡','镍','钠']
jianzhucailiao = ['水泥制造','水泥','玻璃','玻璃制造','管材','耐火材料', '建材']
jianzhuzhuangshi = ['房屋建设','装修装饰','园林工程','基础建设','城轨建设','路桥施工','水利工程','铁路建设','专业工程','钢结构','化学工程','国际工程','家装','光伏建筑']
dianqishebei = ['电机','电网自动化','工控自动化','工控自动化','计量仪表','电源设备','设备','风电','光伏','储能','高低压','高压','低压', '中压','线缆','充电桩','新能源']
jixieshebei = ['通用机械','机床工具','机械','磨具磨料','内燃机','制冷空调','工程机械','重型机械','冶金','楼宇设备','环保设备','纺织服装','农用机械','印刷',
               '仪器', '仪表', '金属制品','铁路设备' ,'运输设备','挖掘机','装载机','工业自动化','智能装备','工业气体']
guofangjungong = ['航天装备','航天','地面兵装','船舶制造','航空装备','军工','国防','海军','陆军','空军','火箭','导弹']
qiche = ['乘用车','汽车','汽车零部件','汽车服务','其他交运服务','新能源汽车','特斯拉']
jiayongdianqi = ['冰箱','空调','洗衣机','小家电','彩电','家电','家用电器','格力','美的','扫地机器人','海尔']
fangzhifuzhuang = ['纺织','毛纺','棉纺','丝绸','印染','辅料','男装','女装','休闲服装','鞋帽','家纺','其他服装','棉纱','坯布','纱线','运动服饰','安踏','申洲国际']
qinggongzhizao = ['造纸','包装','包装印刷','家具','其他家用轻工','珠宝首饰','文娱用品','其他轻工','纸业','文具','公牛']
shangyemaoyi = ['商贸','商业贸易','百货','超市','专业连锁','连锁','贸易','电商','直播带货']
nonglinmaoyu = ['种植业','种子生产','粮食种植','渔业','海洋捕捞','水产','水产养殖','林业','饲料','农村品加工','果蔬加工','农业综合','畜禽','动物','猪肉','大米','小麦','水稻','牛肉','动保','猪瘟']
shipinyinliao = ['饮料','白酒','啤酒','软饮','软饮料','葡萄酒','酒','黄酒','肉制品','调味发酵品','乳制品','食品综合','牛奶','冷链','餐饮']
xiuxianfuwu = ['人工景点','自然景点','景点','自然','酒店','旅游','餐饮','休闲','免税']
yiyaoshengwu = ['医药','生物','化学制药','中药','生物制品','医药商业','医疗器材','医疗服务','医疗器械','血制品','中成药','原料药','CXO ','CDMO']
gongyongshiye = ['公用事业','电力','火电','水电','燃机发电','热电','新能源发电','水务','燃气','环保工程']
jiaotongyunshu = ['交通运输','港口','高速公路','公交','航空运输','机场','航运','铁路运输','物流','快递']
fangdichan = ['房地产开发','园区开发','房地产','房子','房','地产','园区','物业']
dianzi = ['半导体','电子','集成电路','分立器件','元件','电路板','光学','LED','电子制造','电子系统','电子零件','被动元件','PCB','光刻','蚀刻','芯片','制程','苹果','消费电子','手机']
jisuanji = ['计算机','电脑','计算机设备','计算机应用','软件开发','IT','IT服务','计算机应用','服务器','数据中心','网络安全','杀毒软件']
chuanmei = ['传媒','文化传媒','平面媒体','影视','动漫','有线电视','营销传播','营销服务','互联网传媒','移动互联网','互联网信息','其他互联网','游戏','腾讯','互联网','阿里巴巴','广告']
tongxin = ['通信','通信设备','通信运营','通信传输','通信配套','终端设备','传输','5G']
yinhang = ['银行','bank','存款','贷款','存贷比','理财产品','私人银行','资产管理']
feiyinjinrong = ['非银','非银金融','保险','证券','多元金融','综合金融']
zonghe = ['综合','检测']

ttt = [caijue, huagong, gangtie, yousejinshu, jianzhucailiao, jianzhuzhuangshi, dianqishebei, jixieshebei,
       guofangjungong, qiche, jiayongdianqi, fangzhifuzhuang, qinggongzhizao,
       shangyemaoyi, nonglinmaoyu, shipinyinliao, xiuxianfuwu, yiyaoshengwu, gongyongshiye, jiaotongyunshu, fangdichan,
       dianzi, jisuanji, chuanmei, tongxin, yinhang, feiyinjinrong, zonghe]


def moha(t):
    x = abc[abc.keywords.apply(lambda sentence: any(word in sentence for word in t))]
    bbx = x['counts'].sum()
    list1.append(bbx)


for i in ttt:
    moha(i)

aaa = pd.DataFrame(list2)
aab = pd.DataFrame(list1)
aaa = pd.concat([aaa,aab], axis=1)
aaa.columns = ["frequency", "species"]
aaa.to_excel('/Users/19723/Desktop/评论词频.xlsx')
print('Processed finished')