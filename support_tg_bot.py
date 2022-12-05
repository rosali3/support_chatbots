import telebot
import pandas as pd
import re
from nltk import word_tokenize, pos_tag
from nltk.stem import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances

token = '59....' # токен из отца ботов
bot = telebot.TeleBot(token)
df = pd.read_excel(r"dialog_talk_agent.xlsx") # эксель табличка с ответами на вопросы
df.ffill(axis=0, inplace=True) #заполняем пустые строки в таблице
askCount=0
user_id = 0
support_id = 0000000 # мой айди который я узнала из print(message.chat.id)

def text_normalize (text): # функция для нормализации сообщения
    words = str(text).lower() # текст в нижний регистр
    words = re.sub(r'[^a-z 0-9]', ' ', words) # оставляем только нужные символы
    tokens = word_tokenize(words) # токенизируем
    tags = pos_tag(tokens) # определяем часть речи
    lemm_list = [] # пустой список для уже лемматизированных слов
    lemm = wordnet.WordNetLemmatizer() # лемматайзер который будем использовать
    for word, tag in tags:
        if tag.startswith('V'): #меняем одно обозначение части речи на другое
            pos = 'v'
        elif tag.startswith('J'):
            pos = 'a'
        elif tag.startswith('R'):
            pos = 'r'
        else:
            pos = 'n'
        lemm_list.append(lemm.lemmatize(word, pos)) # присоединяем к списку слово и ее часть речи
    return ' '.join(lemm_list)

df['Lemmatize'] = df['Context'].apply(text_normalize) # ?

cv = CountVectorizer() # инструмент который превращает текст в бинарную матрицу 
X = cv.fit_transform(df['Lemmatize']).toarray() # ?

features = cv.get_feature_names_out()
df_bow = pd.DataFrame(X, columns = features)

def delete_stopwords (text):
    stop = stopwords.words('english') # определяем что такое стоп слова
    t = []
    for word in str(text).split(' '):
        if word not in stop:
            t.append(word)
    text = ' '.join(t)
    return text

def give_answer (initial_question):
    question = text_normalize(delete_stopwords(initial_question))
    ques_bow = cv.transform([question]).toarray() # текст в массив
    cos_value = 1 - pairwise_distances(df_bow, ques_bow, metric = 'cosine') # оцениваем косинусное расстояние между матрицей вопроса пользователя и возможными ответами из подготовленной таблицы
    ans_ind = cos_value.argmax() # индекс = косинусное расстояние
    ans = df['Text Response'].loc[ans_ind] # достаем ответ на вопрос из таблички 
    return ans, cos_value, question  # ответ на вопрос из таблички

@bot.message_handler(func=lambda m: True)
# print('запущено')
def echo_all(message):
    global user_id
    global askCount
    global cos_value
    global question

    if(message.chat.id != support_id):
        askCount +=1
        user_id = message.chat.id
        answer = give_answer(message.text)
        bot.reply_to(message, answer)
        summ = 0
        global cos_value
        for znach in cos_value:
            sum += znach[0]
        if sum >=0 and sum < 500:
            ans = "подождите ответа сотрудника поддержки"
            bot.send_message(support_id, question)
        else:
            df['Simil'] = cos_value
            ans_ind = cos_value.argmax()
            ans = df['Text Response'].loc[ans_ind]
        bot.reply_to(message, ans)
    else:
        askCount = 0
        answer = message.text 
        bot.send_message(user_id, f"ответ от сотрудника поддержки:\n{answer}")
bot.infinity_polling()