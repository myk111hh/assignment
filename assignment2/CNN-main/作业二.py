# step 1 加载原数据
jieba.load_userdict(project.aux_dir + dict_path)
data_local_df = pd.read_csv(project.data_dir + train_all, sep='\t', header=None, names=["index", "s1", "s2", "label"])
data_test_df = pd.read_csv(project.data_dir + test_all, sep='\t', header=None, names=["index", "s1", "s2", "label"])
data_all_df = pd.read_csv(project.data_dir + train_data_all, sep='\t', header=None,
                          names=["index", "s1", "s2", "label"])
# 预训练中文字符向量
pre_train_char_w2v()
# 训练集中的语句的文本处理，去除停用词，根据规则表替换相应的词，使用jieba对语句进行分词处理
preprocessing(data_local_df, 'train_0.6_seg')
preprocessing(data_test_df, 'test_0.4_seg')
preprocessing(data_all_df, 'data_all_seg')

# 保存label
project.save(project.features_dir + 'y_0.4_test.pickle', data_test_df['label'].tolist())
project.save(project.features_dir + 'y_0.6_train.pickle', data_local_df['label'].tolist())
project.save(project.features_dir + 'y_train.pickle', data_all_df['label'].tolist())

# step 2预训练中文词组向量
pre_train_w2v()

# step 3记录训练集中的词汇表中的词对应的词向量表示
process_save_embedding_wv('train_all_w2v_embedding_matrix.pickle', type=2, isStore_ids=True)
# process_save_embedding_wv('zhihu_w2v_embedding_matrix.pickle',type=2,isStore_ids=False)
# process_save_embedding_wv('zhihu_w2v_embedding_matrix.pickle',type=3,isStore_ids=False)

# step 4 char wordembedding
process_save_char_embedding_wv(isStore_ids=True)
