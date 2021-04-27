import pandas as pd
from sklearn.tree import DecisionTreeClassifier
video_game_data = pd.read_csv('vgsales.csv')
x = video_game_data.drop(columns=['genre'])
y = video_game_data['genre']
x_train, x_test,y_train, y_test = train_test_spli(x, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
score = accuracy_score(y_test, prediction)
print(score)
