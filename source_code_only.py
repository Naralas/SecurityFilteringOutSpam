import sklearn

from sklearn import datasets
from sklearn import datasets
from sklearn.model_selection import train_test_split

email_set = datasets.load_files("data/")

RATIO_TRAINING = 0.7

X_train, X_test, Y_train, Y_test = train_test_split(email_set.data, email_set.target, 
                                                            test_size=(1-RATIO_TRAINING), shuffle=True)

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


bayes_clf = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])

bayes_clf.fit(X_train, Y_train)
bayes_predicted = bayes_clf.predict(X_test)


from sklearn.metrics import classification_report


print("Naive Bayes")
print("------------------------------------------------------------")
print(classification_report(Y_test, bayes_predicted))


test_data = ['language speaker', 'have sex with ladies', 'student historian', 'explode',
            'Hi, the conference will be held in Bern tomorrow at 12 PM.',
           'Please find attached the project\'s specification. Call me if you need anything.',
           'You need to reset your account at "http://www.bank.russia.po.biz/useraccount"',
           'Earn 3841 dollars a month with this simple trick !',
           'Hot girls waiting for your meat scepter in your area',
           'Enlarge your webgl in 22 weeks with our new feature']


predicted = bayes_clf.predict(test_data)

for doc, category in zip(test_data, predicted):
    print('%r => %s' % (doc, email_set.target_names[category]))