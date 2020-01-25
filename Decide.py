import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

sns.set_style('white')

def isNotValid(code, keys='bpflw'):
    return any(char.isdigit() or (char not in keys) for char in code)


def decide(model_pkl, reviews_csv, args):
    try:
        support_vector_model_pkl = open(model_pkl, 'rb')
    except:
        print("ERROR!! \'" + model_pkl + "\' NOT FOUND")
        return

    try:
        reviews = pd.read_csv(reviews_csv)
    except:
        try:
            reviews = pd.read_csv(reviews_csv, encoding = "ISO-8859-1")
        except:   
            print("ERROR!! \'" + reviews_csv + "\' NOT FOUND")
            return
    

    try:
        model = pickle.load(support_vector_model_pkl)
    except:
        print("ERROR! Make sure the model is written with scikit-learn v0.19.0 or higher")
        return
    
    cv = CountVectorizer()

    try:
        X = reviews['text']
    except:
        print("ERROR! Please Make sure your reviews falls under a label called \'text\'")
        return

    X = cv.fit_transform(X)
    predictions = model.predict(X)
    reviews['start'] = predictions
    reviews.to_csv('Classified_Reviews.csv', index=False)
    reviews['length'] = reviews['text'].apply(len)


    if 'w' in args:
        plt.clf()
        sns.boxplot(x='stars',y='length',data=reviews,palette='rainbow')
        plt.savefig('Input BoxPlot.png')

    
    if 'b' in args:
        plt.clf()
        sns.countplot(x='stars',data=reviews,palette='rainbow')
        plt.savefig('Output BarPlot.png')
        
    if 'l' in args:
        plt.clf()
        g = sns.FacetGrid(reviews,col='stars')
        g.map(plt.hist,'length')
        plt.savefig('Input lengthHistogram.png')
        
    if 'f' in args:
        # awaits implementation
        pass
        
    if 'p' in args:
        # awaits implementation
        pass
        #plt.savefig('Output PiePlot.png')
    

if len(sys.argv) == 4 or len(sys.argv) == 3:
    if (sys.argv[1])[-4:] == '.pkl':
        if (sys.argv[2])[-4:] == '.csv':
            if len(sys.argv) == 4:
                if not isNotValid(sys.argv[3].lower()):
                    decide(sys.argv[1], sys.argv[2], sys.argv[3].lower())
                else:
                    print("ERROR! Please check your visualization's arguments")
                print("Finished!")
            else:
                decide(sys.argv[1], sys.argv[2], '')
                print("Finished!")

        else:
            print("ERROR!\nCommand -->python Decide.py classifier.pkl reviews_file.csv")
    

    else:
        print("ERROR!\nCommand -->python Decide.py classifier.pkl reviews_file.csv")

else:
    print("ERROR!\nCommand -->python Decide.py classifier.pkl reviews_file.csv")
