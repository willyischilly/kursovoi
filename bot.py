#!/usr/bin/env python
# coding: utf-8

import random
import os
import time
import nltk
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc
)

import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import pygame

# 1. НАСТРОЙКИ

BOT_CONFIG = {
    'intents': {
        'hello': {
            'examples': ['Привет', 'Здравствуй', 'Добрый день', 'Хай', 'Хелло', 'Салют', 'Приветствую', 'Здравствуйте'],
            'responses': ['Добрый день! Рад вас видеть!', 'Приветствую! Как ваши дела?', 'Здравствуйте! Чем могу помочь?']
        },
        'bye': {
            'examples': ['Пока', 'До свидания', 'Прощай', 'До встречи', 'Я пошел', 'Счастливо'],
            'responses': ['До свидания! Заходите ещё!', 'Всего доброго!', 'Был рад пообщаться!']
        },
        'name': {
            'examples': ['Как тебя зовут?', 'Кто ты?', 'Твое имя?', 'Представься', 'Как зовут?', 'Как твое имя?', 'Имя?', 'Зовут?'],
            'responses': ['Я TimeBot, ваш виртуальный собеседник.', 'Меня зовут TimeBot. Очень приятно!']
        },
        'how_are_you': {
            'examples': ['Как дела?', 'Как жизнь?', 'Как настроение?', 'Что нового?', 'Как ты?', 'Как сам?', 'Как у тебя дела?', 'Как поживаешь?', 'Как оно?', 'Как твои дела?'],
            'responses': ['У меня всё отлично! А у вас?', 'Всё хорошо, работаю над улучшением сервиса!', 'Прекрасно! Общаюсь с интересными людьми.', 'Спасибо, всё замечательно! Как ваши дела?']
        },
        'fine_and_you': {
            'examples': ['Хорошо как ты', 'Нормально а у тебя', 'Отлично как сам', 'Хорошо как дела', 'Нормально а ты как', 'Отлично а ты', 'Хорошо а ты', 'Нормально как ты'],
            'responses': ['У меня тоже всё хорошо! Чем могу помочь?', 'Рад это слышать! У меня всё отлично.', 'Спасибо, у меня всё прекрасно!']
        },
        'im_fine': {
            'examples': ['У меня хорошо', 'У меня тоже хорошо', 'У меня отлично', 'Тоже хорошо', 'Всё хорошо', 'Нормально', 'У меня нормально', 'У меня всё хорошо'],
            'responses': ['Отлично! Рад за вас!', 'Это замечательно!', 'Прекрасно! Что ещё расскажете?']
        },
        'thanks': {
            'examples': ['Спасибо', 'Благодарю', 'Спасибо большое', 'От души', 'Спасибо за помощь'],
            'responses': ['Пожалуйста! Обращайтесь.', 'Рад помочь!', 'Не стоит благодарности!']
        },
        'weather': {
            'examples': ['Какая погода?', 'Что на улице?', 'Холодно сегодня?', 'Погода'],
            'responses': ['Я не знаю погоду за окном, но хорошие часы всегда подскажут точное время!', 'Погода - штука переменчивая. А вот хорошие часы - это навсегда!']
        },
        'hobby': {
            'examples': ['Чем увлекаешься?', 'Твои хобби?', 'Что любишь?', 'Чем занимаешься?'],
            'responses': ['Я увлекаюсь точными механизмами и коллекционирую знания о часах!', 'Моё хобби - помогать людям выбирать лучшие часы.']
        },
        # ДИАЛОГОВЫЕ НАМЕРЕНИЯ
        'dialog_yes': {
            'examples': ['Да', 'Давай', 'Покажи', 'Показывай', 'Да покажи', 'Да показывай', 'Ага', 'Ну давай', 'Интересно', 'Конечно', 'Да конечно', 'Ок'],
            'responses': None
        },
        'dialog_no': {
            'examples': ['Нет', 'Не надо', 'Не хочу', 'Не сейчас', 'Потом', 'Я просто смотрю', 'Отстань'],
            'responses': None
        },
        # ПОКУПАТЕЛЬСКИЕ НАМЕРЕНИЯ
        'buy_signal': {
            'examples': ['Хочу купить часы', 'Ищу подарок', 'Нужна помощь с выбором часов', 'Посоветуй что купить', 
                         'Что подарить?', 'Ищу часы', 'Нужны часы', 'Присматриваю часы', 'Посоветуй часы для спорта'],
            'responses': None
        },
        'interested_buy': {
            'examples': ['Какие есть модели у вас', 'Покажи ассортимент', 'Что есть в наличии из брендов', 'Мужские модели', 'Женские модели'],
            'responses': [
                'У нас отличная коллекция: 1) Rolex - статус и роскошь. 2) Omega - точность и история. 3) Casio - надёжность и технологии. Что вас интересует?',
                'Вы знаете, у нас есть потрясающие часы! Rolex, Omega и Casio. Какой бренд вам ближе?'
            ]
        },
        'rolex_intent': {
            'examples': ['Расскажи про Rolex', 'Rolex', 'Покажи Rolex', 'Хочу Rolex', 'Ролекс', 'Что за Rolex'],
            'responses': ['Rolex - символ успеха! Daytona (от 1 500 000 руб.) - спортивная классика. Datejust - элегантность на каждый день. Желаете оформить заказ?']
        },
        'omega_intent': {
            'examples': ['Расскажи про Omega', 'Omega', 'Омега', 'Хочу Omega', 'Что за Omega', 'А что за Omega'],
            'responses': ['Omega - выбор астронавтов и агента 007! Speedmaster (от 450 000 руб.) - лунные часы. Seamaster (от 380 000 руб.) - для дайвинга. Заинтересованы?']
        },
        'casio_intent': {
            'examples': ['Расскажи про Casio', 'Casio', 'Касио', 'Джи-шок', 'G-Shock', 'Покажи Casio'],
            'responses': ['Casio - японское качество! G-Shock (от 12 000 руб.) - ударопрочные. Edifice (от 18 000 руб.) - стиль и технологии. Будете брать?']
        },
        'confirm_order': {
            'examples': ['Да оформляй', 'Заказываю', 'Закажи', 'Купить', 'Беру', 'Оформи', 'Да беру', 'Да заказываю', 'Оформляй', 'Бери', 'Давай заказывай', 'Да оформляй g-shock', 'Да бери'],
            'responses': ['Отлично! Детали заказа отправлены менеджеру. С вами свяжутся для уточнения. Спасибо за доверие!']
        },
        'reject_buy': {
            'examples': ['Нет не надо', 'Не интересует', 'Спасибо не сейчас', 'Я просто смотрю', 'Потом', 'Не сейчас', 'Отстань'],
            'responses': ['Хорошо, никаких проблем! Продолжим общение. О чём ещё поговорим?', 'Понимаю! Если что - я всегда здесь. О чём хотите поговорить?']
        },
    },
    'failure_phrases': [
        'Интересная мысль! Расскажите поподробнее?',
        'Я над этим работаю. Давайте поговорим о чём-нибудь ещё?',
        'Не совсем понял, но мне интересно ваше мнение!',
    ],
    'sentiment': {
        'positive': ['люблю', 'отлично', 'прекрасно', 'круто', 'супер', 'великолепно', 'хочу', 'нравится', 'хороший', 'красивый', 'хорошо'],
        'negative': ['отвратительно', 'плохо', 'ужасно', 'не нравится', 'дорого', 'фу', 'не хочу', 'отстой']
    }
}

# 2. ЛЕММАТИЗАЦИЯ
print("Инициализация NLP (Natasha)...")
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

def lemmatize_natasha(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    return [token.lemma for token in doc.tokens if token.lemmatize(morph_vocab) or True]

# 3. ОБРАБОТКА ВВОДА
SPELL_DICT = set()
for intent_data in BOT_CONFIG['intents'].values():
    for example in intent_data['examples']:
        for word in example.split():
            SPELL_DICT.add(word.lower())
SPELL_DICT.update(['rolex', 'omega', 'casio', 'часы', 'купить', 'хочу', 'подарок', 'нужен', 'ищу', 'посоветуй', 'g-shock', 'gshock', 'дела', 'тебя', 'хорошо', 'нормально', 'отлично', 'зовут', 'зовут', 'самые', 'точные', 'спорт', 'спорта'])

def clean_and_spell_correct(phrase):
    phrase = phrase.lower()
    allowed_chars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz -?'
    cleaned = ''.join([ch for ch in phrase if ch in allowed_chars])
    cleaned = cleaned.replace('?', '')
    words = cleaned.split()
    corrected = []
    for word in words:
        if len(word) <= 2:
            corrected.append(word)
            continue
        min_dist, best = 2, word
        for dict_word in SPELL_DICT:
            dist = nltk.edit_distance(word, dict_word)
            if dist < min_dist:
                min_dist, best = dist, dict_word
        corrected.append(best)
    return ' '.join(corrected)

# 4. СЕНТИМЕНТ-АНАЛИЗ
def sentiment_analysis(text):
    pos = neg = 0
    for word in text.lower().split():
        if word in BOT_CONFIG['sentiment']['positive']: pos += 1
        if word in BOT_CONFIG['sentiment']['negative']: neg += 1
    total = pos + neg
    return 0 if total == 0 else (pos - neg) / total

# 5. ML-КЛАССИФИКАЦИЯ
print("Обучение ML-классификатора...")
X_text, y = [], []
for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        X_text.append(example)
        y.append(intent)

vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
X = vectorizer.fit_transform(X_text)
clf = LinearSVC().fit(X, y)
print("ML-классификатор готов!")

def classify_intent(replica):
    replica = clean_and_spell_correct(replica)
    if not replica.strip():
        return None
    try:
        all_examples = []
        for intent_name, intent_data in BOT_CONFIG['intents'].items():
            for ex in intent_data['examples']:
                all_examples.append((intent_name, ex))
        
        min_dist = 100
        best_intent = None
        for intent_name, example in all_examples:
            dist = nltk.edit_distance(replica, clean_and_spell_correct(example))
            if len(example) > 0:
                norm_dist = dist / len(example)
                if norm_dist < min_dist:
                    min_dist = norm_dist
                    best_intent = intent_name
        
        if min_dist < 0.8 and best_intent in BOT_CONFIG['intents']:
            return best_intent
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# 6. ГЕНЕРАЦИЯ ОТВЕТА
def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        responses = BOT_CONFIG['intents'][intent].get('responses')
        if responses:
            return random.choice(responses)
    return None

# 7. ПОИСК ПО ДАТАСЕТУ
print("Загрузка датасета диалогов...")
dialogues_structured_cut = {}
try:
    with open('dialogues1.txt', encoding='utf-8') as f:
        content = f.read()
    dialogues_str = content.split('\n\n')
    dialogues = [d.split('\n')[:2] for d in dialogues_str]
    filtered = []
    seen = set()
    for q, a in dialogues:
        if len(q) < 3 or len(a) < 3: continue
        q = clean_and_spell_correct(q[2:] if q.startswith('- ') else q)
        a = a[2:] if a.startswith('- ') else a
        if q and q not in seen:
            seen.add(q)
            filtered.append((q, a))
    dialogues_structured = {}
    for q, a in filtered:
        for w in set(q.split()):
            if w not in dialogues_structured:
                dialogues_structured[w] = []
            dialogues_structured[w].append((q, a))
    for w in dialogues_structured:
        dialogues_structured[w] = sorted(dialogues_structured[w], key=lambda x: len(x[0]))[:1000]
    dialogues_structured_cut = dialogues_structured
    print(f"Датасет загружен: {len(filtered)} пар")
except FileNotFoundError:
    print("Файл dialogues.txt не найден.")

def generate_answer_from_dialogs(replica):
    if not dialogues_structured_cut:
        return None
    replica = clean_and_spell_correct(replica)
    words = set(replica.split())
    mini = []
    for w in words:
        if w in dialogues_structured_cut:
            mini.extend(dialogues_structured_cut[w])
    unique = []
    seen = set()
    for q, a in mini:
        if q not in seen:
            seen.add(q)
            unique.append((q, a))
    answers = []
    for q, a in unique:
        if not q: continue
        len_diff = abs(len(replica) - len(q)) / len(q)
        if len_diff < 0.3:
            dist = nltk.edit_distance(replica, q) / len(q)
            if dist < 0.25:
                answers.append((dist, a))
    if answers:
        result = min(answers, key=lambda x: x[0])[1]
        print(f"[DATASET] Найден ответ из датасета!")
        return result
    return None

# 8. ГОЛОСОВЫЕ ФУНКЦИИ
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def listen_to_speech():
    with microphone as source:
        print("Говорите...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            return ""
    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        print(f"Вы сказали: {text}")
        return text
    except:
        return ""

def synthesize_speech(text):
    try:
        tts = gTTS(text=text, lang='ru', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.init()
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except:
        pass

# 9. КОНТЕКСТ ДИАЛОГА
user_state = {
    "message_count": 0,
    "ad_phase": 0,
    "ad_suggested": False,
    "last_brand": None
}

BUY_TRIGGERS = ['купить', 'подарок', 'ищу', 'нужен', 'нужны', 'нужна', 
                'выбрать', 'присмотреть', 'бренд', 'марка']

IMPORTANT_INTENTS = ['name', 'hello', 'bye', 'thanks', 'how_are_you', 'fine_and_you', 'im_fine', 'help', 'weather', 'hobby']

def contains_buy_triggers(text):
    words = clean_and_spell_correct(text).split()
    return any(w in BUY_TRIGGERS for w in words)

def handle_advertising_scenario(intent, replica_text):
    """Плавный переход к рекламе часов"""
    global user_state
    
    text = clean_and_spell_correct(replica_text)
    has_triggers = contains_buy_triggers(text)
    
    # ФАЗА 0: ОБЫЧНЫЙ ДИАЛОГ
    if user_state["ad_phase"] == 0:
        if intent in ['buy_signal', 'interested_buy'] or has_triggers:
            user_state["ad_phase"] = 2
            user_state["ad_suggested"] = True
            return get_answer_by_intent('interested_buy')
        
        if intent is None or intent not in IMPORTANT_INTENTS:
            user_state["message_count"] += 1
        
        if user_state["message_count"] >= 4 and not user_state["ad_suggested"]:
            user_state["ad_phase"] = 1
            user_state["ad_suggested"] = True
            return "Слушайте, мы как раз помогаем с выбором отличных часов. Интересно посмотреть?"
        
        return None
    
    # ФАЗА 1: ПРЕДЛОЖЕНИЕ
    if user_state["ad_phase"] == 1:
        if intent in ['dialog_yes', 'buy_signal', 'interested_buy', 'rolex_intent', 'omega_intent', 'casio_intent']:
            user_state["ad_phase"] = 2
            return get_answer_by_intent('interested_buy')
        elif intent in ['reject_buy', 'dialog_no'] or sentiment_analysis(text) < -0.3:
            user_state = {"message_count": 0, "ad_phase": 0, "ad_suggested": True, "last_brand": None}
            return get_answer_by_intent('reject_buy')
        else:
            user_state["ad_phase"] = 2
            return "Давайте покажу! У нас есть: 1) Rolex - статус. 2) Omega - точность. 3) Casio - надёжность. Что интересует?"
    
    # ФАЗА 2: ВЫБОР БРЕНДА
    if user_state["ad_phase"] == 2:
        brand = None
        if intent == 'rolex_intent': brand = "Rolex"
        elif intent == 'omega_intent': brand = "Omega"
        elif intent == 'casio_intent': brand = "Casio"
        
        if brand:
            user_state["last_brand"] = brand
            user_state["ad_phase"] = 3
            if brand == "Rolex": return get_answer_by_intent('rolex_intent')
            elif brand == "Omega": return get_answer_by_intent('omega_intent')
            elif brand == "Casio": return get_answer_by_intent('casio_intent')
        
        if intent in ['dialog_yes', 'confirm_order']:
            return "Отлично! Какой бренд вас интересует: Rolex, Omega или Casio?"
        
        if intent in ['reject_buy', 'dialog_no']:
            user_state = {"message_count": 0, "ad_phase": 0, "ad_suggested": True, "last_brand": None}
            return get_answer_by_intent('reject_buy')
        
        gen = generate_answer_from_dialogs(text)
        if gen:
            return gen
        
        if intent is None:
            return "Давайте выберем бренд! Вас интересует Rolex, Omega или Casio?"
    
    # ФАЗА 3: ПОКАЗ БРЕНДА / ПОДТВЕРЖДЕНИЕ
    if user_state["ad_phase"] == 3:
        brand = None
        if intent == 'rolex_intent': brand = "Rolex"
        elif intent == 'omega_intent': brand = "Omega"
        elif intent == 'casio_intent': brand = "Casio"
        
        if brand:
            user_state["last_brand"] = brand
            if brand == "Rolex": return get_answer_by_intent('rolex_intent')
            elif brand == "Omega": return get_answer_by_intent('omega_intent')
            elif brand == "Casio": return get_answer_by_intent('casio_intent')
        
        if intent in ['confirm_order']:
            user_state = {"message_count": 0, "ad_phase": 0, "ad_suggested": False, "last_brand": None}
            return get_answer_by_intent('confirm_order')
        
        if intent in ['dialog_yes']:
            user_state = {"message_count": 0, "ad_phase": 0, "ad_suggested": False, "last_brand": None}
            return "Отлично! Сейчас оформлю. " + get_answer_by_intent('confirm_order')
        
        if intent in ['reject_buy', 'dialog_no']:
            user_state = {"message_count": 0, "ad_phase": 0, "ad_suggested": True, "last_brand": None}
            return get_answer_by_intent('reject_buy')
        
        if intent in ['buy_signal', 'interested_buy']:
            user_state["ad_phase"] = 2
            return get_answer_by_intent('interested_buy')
        
        gen = generate_answer_from_dialogs(text)
        if gen:
            return gen
        
        if intent is None:
            return "Я вас не совсем понял. Хотите оформить заказ на " + str(user_state.get("last_brand", "часы")) + "? Или посмотрим другой бренд (Rolex/Omega/Casio)?"
    
    return None

# 10. ОСНОВНАЯ ФУНКЦИЯ БОТА
def bot(replica, use_voice_answer=False):
    if not replica.strip():
        return "Молчание - золото!"
    
    cleaned = clean_and_spell_correct(replica)
    sent = sentiment_analysis(cleaned)
    intent = classify_intent(cleaned)
    print(f"[DEBUG] Фаза: {user_state['ad_phase']} | Intent: {intent} | Счётчик: {user_state['message_count']} | Текст: {cleaned}")
    
    # Сначала проверяем датасет (до сценариев)
    gen_answer = generate_answer_from_dialogs(cleaned)
    
    ad_answer = handle_advertising_scenario(intent, cleaned)
    if ad_answer:
        if use_voice_answer: synthesize_speech(ad_answer)
        return ad_answer
    
    # Если фаза 0 и есть ответ из датасета — возвращаем его
    if user_state["ad_phase"] == 0 and gen_answer:
        if use_voice_answer: synthesize_speech(gen_answer)
        return gen_answer
    
    answer = get_answer_by_intent(intent)
    if answer:
        if use_voice_answer: synthesize_speech(answer)
        return answer
    
    if gen_answer:
        if use_voice_answer: synthesize_speech(gen_answer)
        return gen_answer
    
    failure = random.choice(BOT_CONFIG['failure_phrases'])
    if use_voice_answer: synthesize_speech(failure)
    return failure

# 11. TELEGRAM
TELEGRAM_AVAILABLE = False
try:
    from telegram import Update
    from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
    TELEGRAM_AVAILABLE = True
    print("Telegram-библиотека загружена!")
except ImportError:
    print("Telegram недоступен.")

def start_command(update: Update, context: CallbackContext):
    global user_state
    user_state = {"message_count": 0, "ad_phase": 0, "ad_suggested": False, "last_brand": None}
    update.message.reply_text('Привет! Я TimeBot. Могу поболтать на любые темы, а если захотите - помогу выбрать отличные часы!')

def help_command(update: Update, context: CallbackContext):
    update.message.reply_text('Я бот-собеседник и консультант по часам. Просто общайтесь со мной! Могу рассказать о Rolex, Omega и Casio.')

def handle_message(update: Update, context: CallbackContext):
    reply = bot(update.message.text)
    update.message.reply_text(reply)

def handle_voice(update: Update, context: CallbackContext):
    update.message.reply_text("Голосовые пока не поддерживаются. Напишите текст!")

def main_telegram():
    TOKEN = "8689252024:AAEDbARA0Lc7AqHfRk4jcOBEioALpHz0LDc"
    
    import socks, socket
    try:
        socks.set_default_proxy(socks.SOCKS5, '127.0.0.1', 10808)
        socket.socket = socks.socksocket
        print("Прокси настроен")
    except:
        pass
    
    try:
        updater = Updater(token=TOKEN, use_context=True)
        dispatcher = updater.dispatcher
        dispatcher.add_handler(CommandHandler("start", start_command))
        dispatcher.add_handler(CommandHandler("help", help_command))
        dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
        dispatcher.add_handler(MessageHandler(Filters.voice, handle_voice))
        print("=" * 40)
        print(" Telegram бот запущен!")
        print("=" * 40)
        updater.start_polling()
        updater.idle()
    except Exception as e:
        print(f"Ошибка Telegram: {e}")

#  12. ЗАПУСК 
if __name__ == "__main__":
    print("=" * 40)
    print(" TimeBot: Собеседник + Консультант ")
    print("=" * 40)
    print("1. Консольный режим (текст)")
    print("2. Консольный режим (голос)")
    print("3. Telegram бот")
    
    choice = input("Выберите режим (1-3): ").strip()
    
    if choice == "3":
        if TELEGRAM_AVAILABLE:
            main_telegram()
        else:
            print("Установите: pip install python-telegram-bot PySocks")
    else:
        use_voice = (choice == "2")
        print("\nОбщайтесь на любые темы! Бот плавно предложит часы.\n")
        
        while True:
            if use_voice:
                ui = input("Enter для голоса или текст: ")
                if ui == "":
                    ui = listen_to_speech()
                    if not ui: continue
            else:
                ui = input("Вы: ")
                
            if ui.lower() in ['выход', 'exit', 'quit']:
                print("До свидания!")
                break
                
            print("Бот:", bot(ui, use_voice_answer=use_voice))