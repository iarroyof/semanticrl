

from sklearn.datasets import fetch_20newsgroups
from string import punctuation
import pandas as pd
import re


def clean_doc(document,
              pattern=r'(\S*@\S*\s?)|([A-Z][a-z' + punctuation + ']+: +\w+)'):
    return ' '.join([s for s in document.split('\n')
                        if re.findall(pattern, s) == [] and s != ''])

sim_cats = ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware', 'comp.windows.x',]  # Very similar topics
dis_cats = ['talk.politics.guns', 'sci.med', 'sci.space', 'rec.sport.hockey',
            'soc.religion.christian', 'misc.forsale', 'rec.autos'
            ]  # Very dissimilar topics
# Take train and test data (not suffled according to corpus documentation)
newsgroups_train = fetch_20newsgroups(subset='train',
                                        categories=sim_cats + dis_cats,
                                        #remove=('headers', 'footers', 'quotes')
                                        )
newsgroups_test = fetch_20newsgroups(subset='test',
                                        categories=sim_cats + dis_cats,
                                        #remove=('headers', 'footers', 'quotes')
                                        )                                        
# Create cat --> label map
literal_train_target = [newsgroups_train['target_names'][i]
                    for i in newsgroups_train['target']]
# Create data frame to filter output files
train_df = pd.DataFrame({'text': newsgroups_train.data,
                        'target': literal_train_target,
                        'label': newsgroups_train.target })
print("Number of train documents: %d" % len(train_df.index))
literal_test_target = [newsgroups_test['target_names'][i]
                    for i in newsgroups_test['target']]
test_df = pd.DataFrame({'text': newsgroups_test.data,
                        'target': literal_test_target,
                        'label': newsgroups_test.target })
print("Number of test documents: %d" % len(test_df.index))
#train_df['text'] = train_df['text'].apply(clean_doc)
with open("sim_train.txt", 'w') as f:
    #for d in train_df['text'].apply(clean_doc):
    for d in train_df['text'].apply(clean_doc)[train_df['target'].isin(sim_cats)]:
        f.write(d if d.endswith('\n') else d + '\n')

with open("sim_test.txt", 'w') as f:
    for d in test_df['text'].apply(clean_doc)[test_df['target'].isin(sim_cats)]:
        f.write(d if d.endswith('\n') else d + '\n')

with open("dis_train.txt", 'w') as f:
    for d in train_df['text'].apply(clean_doc)[train_df['target'].isin(dis_cats)]:
        f.write(d if d.endswith('\n') else d + '\n')

with open("dis_test.txt", 'w') as f:
    for d in test_df['text'].apply(clean_doc)[test_df['target'].isin(dis_cats)]:
        f.write(d if d.endswith('\n') else d + '\n')