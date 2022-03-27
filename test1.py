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




def extract_keyword(content):  # 提取关键词
    # 正则过滤 html 标签
    re_exp = re.compile(r'(<style>.*?</style>)|(<[^>]+>)', re.S)
    content = re_exp.sub(' ', content)
    # html 转义符实体化
    content = html.unescape(content)
    # 切割
    seg = [i for i in jieba.cut(content, cut_all=True) if i != '']
    # 提取关键词
    keywords = jieba.analyse.extract_tags("|".join(seg), topK=200, withWeight=False)
    return keywords


excelFile1 = r'/Users/19723/Desktop/今日话题 - 雪球.xlsx'
# excelFile2 = r'/Users/19723/Desktop/财联社A股24小时电报-上市公司动态-今日股市行情报道.xlsx'
# excelFile3 = r'/Users/19723/Desktop/7_24小时全球财经直播_同花顺财经.xlsx'
df1 = pd.DataFrame(pd.read_excel(excelFile1))
# df2 = pd.DataFrame(pd.read_excel(excelFile2))
# df3 = pd.DataFrame(pd.read_excel(excelFile3))
# df4 = pd.concat([df1,df2,df3],axis=0)
df5 = df1.sort_values(['字段'], axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
df5.to_excel(datetime.datetime.now().strftime("%m-%d")+'每日财经.xls')
df5.to_csv(datetime.datetime.now().strftime("%m-%d")+'每日财经.csv')

data = xlrd.open_workbook(datetime.datetime.now().strftime("%m-%d")+'每日财经.xls')
st = data.sheets()[0] #读取第一个表
st1 = st.col_values(3) #读取第二列

t = open(datetime.datetime.now().strftime("%m-%d")+'每日财经.txt','w', encoding='UTF-8')
for i in st1:
    t.write(str(i)+'\n')
t.close()

# 读取文件
fn = open(datetime.datetime.now().strftime("%m-%d")+'每日财经.txt', encoding='UTF-8') # 打开文件
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
word_counts_top200 = word_counts.most_common(100) # 获取前10最高频的词
print(word_counts_top200) # 输出检查

# 词频展示
mask = np.array(Image.open('/Users/19723/Desktop/Everyday财经/SOURCE/gold.png')) # 定义词频背景
wc = wordcloud.WordCloud(
    background_color='white', # 设置背景颜色
    font_path='/Users/19723/Desktop/Everyday财经/SOURCE/Hiragino Sans GB.ttc', # 设置字体格式
    mask=mask, # 设置背景图
    max_words=100, # 最多显示词数
    max_font_size=100 , # 字体最大值
    scale=50  # 调整图片清晰度，值越大越清楚
)

wc.generate_from_frequencies(word_counts) # 从字典生成词云
image_colors = wordcloud.ImageColorGenerator(mask) # 从背景图建立颜色方案
wc.recolor(color_func=image_colors) # 将词云颜色设置为背景图方案
wc.to_file("词频.png") # 将图片输出为文件
plt.imshow(wc) # 显示词云
plt.axis('off') # 关闭坐标轴
plt.show() # 显示图像

df = pd.read_excel('/Users/19723/Desktop/今日话题 - 雪球.xlsx', header=0)
df.columns = ['id', 'title', 'abstract']
df.to_excel('/Users/19723/Desktop/今日话题 - 雪球.xlsx')

# !/usr/bin/python
# coding=utf-8
# 采用TF-IDF方法提取文本关键词
# http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting


"""
       TF-IDF权重：
           1、CountVectorizer 构建词频矩阵
           2、TfidfTransformer 构建tfidf权值计算
           3、文本的关键字
           4、对应的tfidf矩阵
"""


# 数据预处理操作：分词，去停用词，词性筛选
def dataPrepos(text, stopkey):
    l = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']  # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词
    for i in seg:
        if i.word not in stopkey and i.flag in pos:  # 去停用词 + 词性筛选
            l.append(i.word)
    return l


# tf-idf获取文本top10关键词
def getKeywords_tfidf(data, stopkey, topK):
    idList, titleList, abstractList = data['id'], data['title'], data['abstract']
    corpus = []  # 将所有文档输出到一个list中，一行就是一个文档
    for index in range(len(idList)):
        text = '%s。%s' % (titleList[index], abstractList[index])  # 拼接标题和摘要
        text = dataPrepos(text, stopkey)  # 文本预处理
        text = " ".join(text)  # 连接成字符串，空格分隔
        corpus.append(text)

    # 1、构建词频矩阵，将文本中的词语转换成词频矩阵
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)  # 词频矩阵,a[i][j]:表示j词在第i个文本中的词频
    # 2、统计每个词的tf-idf权值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    # 3、获取词袋模型中的关键词
    word = vectorizer.get_feature_names()
    # 4、获取tf-idf矩阵，a[i][j]表示j词在i篇文本中的tf-idf权重
    weight = tfidf.toarray()
    # 5、打印词语权重
    ids, titles, keys = [], [], []
    for i in range(len(weight)):
        print(u"-------这里输出第", i + 1, u"行文本的词语tf-idf------")
        ids.append(idList[i])
        titles.append(titleList[i])
        df_word, df_weight = [], []  # 当前文章的所有词汇列表、词汇对应权重列表
        for j in range(len(word)):
            print(word[j], weight[i][j])
            df_word.append(word[j])
            df_weight.append(weight[i][j])
        df_word = pd.DataFrame(df_word, columns=['word'])
        df_weight = pd.DataFrame(df_weight, columns=['weight'])
        word_weight = pd.concat([df_word, df_weight], axis=1)  # 拼接词汇列表和权重列表
        word_weight = word_weight.sort_values(by="weight", ascending=False)  # 按照权重值降序排列
        keyword = np.array(word_weight['word'])  # 选择词汇列并转成数组格式
        word_split = [keyword[x] for x in range(0, topK)]  # 抽取前topK个词汇作为关键词
        word_split = " ".join(word_split)
        keys.append(word_split)

    result = pd.DataFrame({"id": ids, "title": titles, "key": keys}, columns=['id', 'title', 'key'])
    #         result = pd.DataFrame({"id": ids, "key": keys},columns=['id', 'key'])
    return result


def main():
    # 读取数据集
    dataFile = r'/Users/19723/Desktop/今日话题 - 雪球.xlsx'
    data = pd.read_excel(dataFile)
    # 停用词表
    stopkey = [w.strip() for w in codecs.open('/Users/19723/Desktop/Everyday财经/SOURCE/chinese_stopword.txt', 'r',
                                              encoding='UTF-8').readlines()]
    # tf-idf关键词抽取
    result = getKeywords_tfidf(data, stopkey, 7)
    result.to_csv("/Users/19723/Desktop/Everyday财经/SOURCE/keys_TFIDF.csv", index=False, encoding='UTF-8')
    outcome = pd.DataFrame(result)


if __name__ == '__main__':
    main()



# 替代掉换行符，筛选一波没用的词
# 词频分类新闻
process = pd.read_csv("/Users/19723/Desktop/Everyday财经/SOURCE/keys_TFIDF.csv")
origin = pd.read_excel('/Users/19723/Desktop/今日话题 - 雪球.xlsx')
origin = origin.assign(keys = process['key'])
origin.to_csv('/Users/19723//Desktop/Everyday财经/SOURCE/tmp.csv')
species = pd.read_excel('/Users/19723/Desktop/Everyday财经/SOURCE/种类.xlsx')

maxmi = []

# 对于每条新闻进行分类，并保存jaccard数据至列表
a = 0
for row in origin.itertuples():
    keywords_x = getattr(row, 'keys')
    locals()['list' + str(int(a))] = []
    for row in species.itertuples():
        keywords_y = getattr(row, 'species')  # 输出每一行
        intersection = len(list(set(keywords_x).intersection(set(keywords_y))))
        union = len(list(set(keywords_x).union(set(keywords_y))))
        # 除零处理
        sim = float(intersection) / union if union != 0 else 0
        locals()['list' + str(int(a))].append(float(sim))
    temp = locals()['list' + str(int(a))].index(max(locals()['list' + str(int(a))]))  # 返回最大值所在的索引值
    maxmi.append(temp)
    a += 1

# 对maxmi这个rank进行更高级的处理
maxmii = []
YIQING = 0
YIJISHICHANG = 0
GUONEIGUSHI = 0
HONGGUAN = 0
SHEHUIXINWEN = 0
QIHUO = 0
WAIHUI = 0
FIS = 0
GONGNONG = 0
YILIAO = 0
XIAOFEI = 0
JIAOYU = 0
KEJI = 0
XINNENGYUAN = 0
JIJIAN = 0
SHUZIHUOBI = 0
ZIRANQIHOU = 0
for i in maxmi:
    if i == 0:
        maxmii.append('疫情')
        YIQING += 1
    elif i == 1:
        maxmii.append('一级市场')
        YIJISHICHANG += 1
    elif i == 2:
        maxmii.append('国内股市')
        GUONEIGUSHI += 1
    elif i == 3:
        maxmii.append('宏观')
        HONGGUAN += 1
    elif i == 4:
        maxmii.append('社会新闻')
        SHEHUIXINWEN += 1
    elif i == 5:
        maxmii.append('期货')
        QIHUO += 1
    elif i == 6:
        maxmii.append('外汇')
        WAIHUI += 1
    elif i == 7:
        maxmii.append('FIs')
        FIS += 1
    elif i == 8:
        maxmii.append('工农实业')
        GONGNONG += 1
    elif i == 9:
        maxmii.append('医疗')
        YILIAO += 1
    elif i == 10:
        maxmii.append('消费')
        XIAOFEI += 1
    elif i == 11:
        maxmii.append('教育')
        JIAOYU += 1
    elif i == 12:
        maxmii.append('科技')
        KEJI += 1
    elif i == 13:
        maxmii.append('新能源汽车')
        XINNENGYUAN += 1
    elif i == 14:
        maxmii.append('基建')
        JIJIAN += 1
    elif i == 15:
        maxmii.append('数字货币')
        SHUZIHUOBI += 1
    else:
        maxmii.append('自然气候')
        ZIRANQIHOU += 1

# rank 进行联立
temp = pd.DataFrame(maxmii)

# for i in range(0, 6):
#     b2 = str2.find('，报')
#     b3 = str2.find('点。')
#     str2 = str2[:b2] + '；' + str2[b3+2:] #替换点数

# 去掉括号
origin = origin.assign(rank=temp[0])
origin1 = origin['abstract'].str.extract(r'】(.*)', expand=False)
origin1 = pd.DataFrame(origin1)
origin1.fillna(origin, inplace=True)
origin1 = origin1.assign(rank=origin['rank'])
origin1.to_excel('/Users/19723/Desktop/最终文件.xlsx')

print("yiqing",YIQING
,"yijishichang",YIJISHICHANG
,"GUONEIGUSHI",GUONEIGUSHI
,"HONGGUAN", HONGGUAN
,"SHEHUIXINWEN", SHEHUIXINWEN
,"QIHUO", QIHUO
,"WAIHUI", WAIHUI
,"FIS", FIS
,"GONGNONG", GONGNONG
,"YILIAO", YILIAO
,"XIAOFEI", XIAOFEI
,"JIAOYU", JIAOYU
,"KEJI", KEJI
,"XINNENGYUAN", XINNENGYUAN
,"JIJIAN", JIJIAN
,"SHUZIHUOBI", SHUZIHUOBI
,"ZIRANQIHOU", ZIRANQIHOU)
